import torch
import functools
import packaging.version
from dataclasses import dataclass
import logging

import sfast
from sfast.jit import passes
from sfast.jit.trace_helper import (lazy_trace, apply_auto_trace_compiler)
from sfast.jit import utils as jit_utils
from sfast.cuda.graphs import make_dynamic_graphed_callable
from sfast.utils.gpu_device import device_has_capability, device_has_tensor_core
from sfast.utils.memory_format import apply_memory_format

logger = logging.getLogger()
logging.basicConfig(level = logging.INFO)

class CompilationConfig:

    @dataclass
    class Default:
        '''
        Default compilation config

        memory_format:
            channels_last if tensor core is available, otherwise contiguous_format.
            On GPUs with tensor core, channels_last is faster
        enable_jit:
            Whether to enable JIT, most optimizations are done with JIT
        enable_jit_freeze:
            Whether to freeze the model after JIT tracing.
            Freezing the model will enable us to optimize the model further.
        preserve_parameters:
            Whether to preserve parameters when freezing the model.
            If True, parameters will be preserved, but the model will be a bit slower.
            If False, parameters will be marked as constants, and the model will be faster.
            However, if parameters are not preserved, LoRA cannot be switched dynamically.
        enable_cnn_optimization:
            Whether to enable CNN optimization by fusion.
        enable_fused_linear_geglu:
            Whether to enable fused Linear-GEGLU kernel.
            It uses fp16 for accumulation, so could cause **quality degradation**.
        prefer_lowp_gemm:
            Whether to prefer low-precision GEMM and a series of fusion optimizations.
            This will make the model faster, but may cause numerical issues.
            These use fp16 for accumulation, so could cause **quality degradation**.
        enable_cuda_graph:
            Whether to enable CUDA graph. CUDA Graph will significantly speed up the model,
            by reducing the overhead of CUDA kernel launch, memory allocation, etc.
            However, it will also increase the memory usage.
            Our implementation of CUDA graph supports dynamic shape by caching graphs of
            different shapes.
        enable_triton:
            Whether to enable Triton generated CUDA kernels.
            Triton generated CUDA kernels are faster than PyTorch's CUDA kernels.
            However, Triton has a lot of bugs, and can increase the CPU overhead,
            though the overhead can be reduced by enabling CUDA graph.
        '''
        memory_format: torch.memory_format = (
            torch.channels_last if device_has_tensor_core() else
            torch.contiguous_format)
        enable_jit: bool = True
        enable_jit_freeze: bool = True
        preserve_parameters: bool = False
        enable_cnn_optimization: bool = device_has_tensor_core()
        enable_fused_linear_geglu: bool = device_has_capability(
            8, 0)
        prefer_lowp_gemm: bool = True
        enable_flash_attention: bool = True
        enable_cuda_graph: bool = True
        enable_triton: bool = True


def quantize(model):
    
    def quantize_linear(m):
        # from diffusers.utils import USE_PEFT_BACKEND
        # assert USE_PEFT_BACKEND
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                    dtype=torch.qint8,
                                                    inplace=True)
        return m

    for idx, (module_name, children) in enumerate(model.named_children()):
        quantized_module = quantize_linear(getattr(model, module_name))
        setattr(model, module_name, quantized_module)
    
    
def _enable_flash_attention(use_sage_attention=True):
    if packaging.version.parse(
            torch.__version__) >= packaging.version.parse('2.0.0'):
        if device_has_capability(8, 0):
            logger.info(
                'spda with flash attention is available on transfomer encoder layer'
            )
            # To do: optimize decoder layer
        else:
            logger.info(
                'spda with efficient memory is available on transfomer encoder layer'
            )
    else:
        logger.warning(
            'spda with efficient implementation is not available.'
        )


def _modify_model(
    m,
    enable_cnn_optimization=True,
    enable_fused_linear_geglu=True,
    prefer_lowp_gemm=True,
    enable_triton=False,
    enable_triton_reshape=False,
    enable_triton_layer_norm=False,
    memory_format=None,
):
    if enable_triton:
        from sfast.jit.passes import triton_passes

    training = getattr(m, 'training', False)

    torch._C._jit_pass_inline(m.graph)

    # sfast._C._jit_pass_erase_scalar_tensors(m.graph)
    sfast._C._jit_pass_eliminate_simple_arith(m.graph)

    # passes.jit_pass_prefer_tanh_approx_gelu(m.graph)

    if not training:
        passes.jit_pass_remove_dropout(m.graph)

    passes.jit_pass_remove_contiguous(m.graph)
    passes.jit_pass_replace_view_with_reshape(m.graph)
    if enable_triton:
        if enable_triton_reshape:
            triton_passes.jit_pass_optimize_reshape(m.graph)

        # triton_passes.jit_pass_optimize_cnn(m.graph)

        triton_passes.jit_pass_fuse_group_norm_silu(m.graph)
        triton_passes.jit_pass_optimize_group_norm(m.graph)

        if enable_triton_layer_norm:
            triton_passes.jit_pass_optimize_layer_norm(m.graph)

    if enable_fused_linear_geglu and not training:
        passes.jit_pass_fuse_linear_geglu(m.graph)

    if not training:
        passes.jit_pass_optimize_linear(m.graph)

    if memory_format is not None:
        sfast._C._jit_pass_convert_op_input_tensors(
            m.graph,
            'aten::_convolution',
            indices=[0],
            memory_format=memory_format)

    if enable_cnn_optimization:
        passes.jit_pass_optimize_cnn(m.graph)

    if prefer_lowp_gemm and not training:
        passes.jit_pass_prefer_lowp_gemm(m.graph)
        passes.jit_pass_fuse_lowp_linear_add(m.graph)
    

def _ts_compiler(
    m,
    inputs,
    modify_model_fn=None,
    freeze=False,
    preserve_parameters=False,
):
    with torch.jit.optimized_execution(True):
        if freeze and not getattr(m, 'training', False):
            # raw freeze causes Tensor reference leak
            # because the constant Tensors in the GraphFunction of
            # the compilation unit are never freed.
            m = jit_utils.better_freeze(
                m,
                preserve_parameters=preserve_parameters,
            )

        if modify_model_fn is not None:
            modify_model_fn(m)

    return m


def _build_ts_compiler(config,
                    enable_triton_reshape=False,
                    enable_triton_layer_norm=False):
    modify_model = functools.partial(
        _modify_model,
        enable_cnn_optimization=config.enable_cnn_optimization,
        enable_fused_linear_geglu=config.enable_fused_linear_geglu,
        prefer_lowp_gemm=config.prefer_lowp_gemm,
        enable_triton=config.enable_triton,
        enable_triton_reshape=enable_triton_reshape,
        enable_triton_layer_norm=enable_triton_layer_norm,
        memory_format=config.memory_format,
    )

    ts_compiler = functools.partial(
        _ts_compiler,
        freeze=config.enable_jit_freeze,
        preserve_parameters=config.preserve_parameters,
        modify_model_fn=modify_model,
    )

    return ts_compiler


def _build_lazy_trace(config,
                    enable_triton_reshape=False,
                    enable_triton_layer_norm=False):

    lazy_trace_ = functools.partial(
        lazy_trace,
        ts_compiler=_build_ts_compiler(
            config,
            enable_triton_reshape=enable_triton_reshape,
            enable_triton_layer_norm=enable_triton_layer_norm),
        check_trace=False,
        strict=False,
    )

    return lazy_trace_


def _jit_optimize(m, config, need_lazy_trace=False, enable_cuda_graph=True):
    if need_lazy_trace:
        lazy_trace_ = _build_lazy_trace(
            config,
            enable_triton_reshape=enable_cuda_graph,
            enable_triton_layer_norm=enable_cuda_graph,
        )
        m.forward = lazy_trace_(m.forward)
    else:
        ts_compiler = _build_ts_compiler(
            config,
            enable_triton_reshape=enable_cuda_graph,
            enable_triton_layer_norm=enable_cuda_graph,
        )
        m = apply_auto_trace_compiler(m, ts_compiler=ts_compiler)
    return m

       
def compile_base_architecture(m, config):
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'
    
    if config.enable_flash_attention:
        _enable_flash_attention()

    if config.memory_format is not None:
        apply_memory_format(m, memory_format=config.memory_format)
        
    if enable_cuda_graph:
        m.forward = make_dynamic_graphed_callable(m.forward)

    if config.enable_jit:
        m = _jit_optimize(m, config, True, enable_cuda_graph)
    

def compile_main_task(m, config):
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'
    
    if config.enable_flash_attention:
        _enable_flash_attention()

    if config.memory_format is not None:
        apply_memory_format(m, memory_format=config.memory_format)

    if enable_cuda_graph:
        m.forward = make_dynamic_graphed_callable(m.forward)

         
def compile_print_reco_model(model, config):
    # for (name, children) in model.named_children():
    #     print(1,name)
    #     for n, c in children.named_children():
    #         print(2,n)
    #         for nn, cc in c.named_children():
    #             print(3,nn)
    quantize(model)
    compile_base_architecture(model.base_architecture, config)
    compile_main_task(model.main_task, config)
    return model