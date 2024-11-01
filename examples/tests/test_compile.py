import functools
import torch
import logging
import packaging.version
from dataclasses import dataclass
import time
import torch.nn.functional as F
import io
import sfast
from sfast.jit import passes
from sfast.jit.trace_helper import (lazy_trace, apply_auto_trace_compiler)
from sfast.jit import utils as jit_utils
from sfast.cuda.graphs import (
    make_dynamic_graphed_callable,
    # apply_auto_graph_compiler,
)
from sfast.utils import gpu_device
from sfast.utils.memory_format import apply_memory_format
from sfast.utils.image_utils import load_image, pil_to_numpy, numpy_to_pt, resize

logger = logging.getLogger()
logging.basicConfig(level = logging.INFO)

SOURCE_DIR = "/root/autodl-tmp/data/"
TRT_PATH = SOURCE_DIR + 'prt_ft13_emptyrecog_invert_0411_147_opt.ts'
PATH = SOURCE_DIR + 'uniocr_wfeat_240919134829_uniocr_ft47xkat6_240903_75.pth'
IMG_URL = SOURCE_DIR + 'text_rectification.jpg'
TRT_SO_PATH = SOURCE_DIR + 'trt_executor_extension_py38.so'

def device_has_tensor_core():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major >= 7
    return False


def device_has_capability(major, minor):
    if torch.cuda.is_available():
        major_, minor_ = torch.cuda.get_device_capability()
        return (major_, minor_) >= (major, minor)
    return False


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
        trace_scheduler:
            Whether to trace the scheduler.
        '''
        memory_format: torch.memory_format = (
            torch.channels_last if device_has_tensor_core() else
            torch.contiguous_format)
        enable_jit: bool = True
        enable_jit_freeze: bool = True
        preserve_parameters: bool = True
        enable_cnn_optimization: bool = device_has_tensor_core()
        enable_fused_linear_geglu: bool = device_has_capability(
            8, 0)
        prefer_lowp_gemm: bool = True
        enable_flash_attention: bool = True
        enable_cuda_graph: bool = True
        enable_triton: bool = False
        trace_scheduler: bool = False


class IterationProfiler:

    def __init__(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0

    def get_iter_per_sec(self):
        if self.begin is None or self.end is None:
            return None
        self.end.synchronize()
        dur = self.begin.elapsed_time(self.end)
        return self.num_iterations / dur * 1000.0

    def callback_on_step_end(self, pipe, i, t, callback_kwargs):
        if self.begin is None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.begin = event
        else:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.end = event
            self.num_iterations += 1
        return callback_kwargs

def hash_arg(arg):
    # micro optimization: bool obj is an instance of int
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(((hash_arg(k), hash_arg(v)) for k, v in arg.items()),
                key=lambda x: x[0]))
    return type(arg)

def compile(model, config):
        
    def quantize_linear(m):
        # from diffusers.utils import USE_PEFT_BACKEND
        # assert USE_PEFT_BACKEND
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                    dtype=torch.qint8,
                                                    inplace=True)
        return m

    def apply_memory_format(m, memory_format=torch.preserve_format):
        
        def convert(t):
            if memory_format is not None and t.dim() in (4, 5) and not (
                    memory_format == torch.channels_last and t.dim() != 4):
                return t.to(memory_format=memory_format)
            return t

        return m._apply(convert)

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

        sfast._C._jit_pass_erase_scalar_tensors(m.graph)
        sfast._C._jit_pass_eliminate_simple_arith(m.graph)

        passes.jit_pass_prefer_tanh_approx_gelu(m.graph)

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

    def _jit_optimize(m, need_lazy_trace=False, enable_cuda_graph=True):
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
            m = _jit_optimize(m, True, enable_cuda_graph)
        
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
        
    for idx, (module_name, children) in enumerate(model.named_children()):
        quantized_module = quantize_linear(getattr(model, module_name))
        setattr(model, module_name, quantized_module)
        
    compile_base_architecture(model.base_architecture, config)
    compile_main_task(model.main_task, config)

def prepare_input(img_url, device, batch=1, height = 64, width = 832, pad_value = 255.0, dtype = torch.float16, interpolation = "bilinear", is_vertical=False):
    padding_left_, padding_top_, padding_right_, padding_bottom_ = 0, 0, 0, 0
    min_resize_ratio_inv = 0.3
    
    image = load_image(img_url)
    img_arr = pil_to_numpy(image)
    _, height_, width_, _ = img_arr.shape
    
    output_height = height
    output_width = width
    if is_vertical:
        resize_ratio_ = max(float(width_) / output_width, min_resize_ratio_inv)
        output_height_ = min(int(height_ / resize_ratio_), output_height)
        # 竖排文字底部填充
        padding_bottom_ = max(0, output_height - output_height_)
        # 竖排文字可以左右填充
        output_width_ = min(int(width_ / resize_ratio_), output_width)
        padding_left_ = max(0, (output_width - output_width_) // 2)
        padding_right_ = max(0, output_width - output_width_ - padding_left_)

    else:
        resize_ratio_ = max(float(height_) / output_height, min_resize_ratio_inv)
        output_width_ = min(int(width_ / resize_ratio_), output_width)
        # 横排文字右侧填充
        padding_right_ = max(0, output_width - output_width_)
        # 横排文字可以上下填充
        output_height_ = min(int(height_ / resize_ratio_), output_height)
        padding_top_ = max(0, (output_height - output_height_) // 2)
        padding_bottom_ = max(0, output_height - output_height_ - padding_top_)


    image_t = numpy_to_pt(img_arr)
    image_t = F.interpolate(image_t, (output_height_, output_width_), mode=interpolation, align_corners=True)
    if interpolation == 'bicubic':
        image_t = image_t.clamp(0, 255)
    image_t = F.pad(
        image_t, (padding_left_, padding_right_, padding_top_, padding_bottom_), value=pad_value
    )
    image_t = image_t.to(device=device, dtype=dtype)

    mask_t = torch.ones((batch, 1, output_height, output_width), device=device, dtype=dtype)[:, 0]
    token_ids_t = torch.ones((image_t.shape[0], 1), device=device, dtype=torch.int64)
    return image_t, mask_t, token_ids_t
    
def main():
    print(torch.__version__)
    torch.classes.load_library(TRT_SO_PATH)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(PATH, map_location=device)
    model.eval()
    
    images_t, masks_t, token_ids_t = prepare_input(img_url=IMG_URL, device=device)
    """
    not compile
    """
    begin = time.time()
    output = trt_model(images_t, masks_t, token_ids_t)
    end = time.time()
    
    print(f'Inference time: {end - begin:.3f}s')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
    
    """
    compile
    """
    config = CompilationConfig.Default()
    compile(model, config)

    print('Begin warmup')
    for _ in range(3):
        model(images_t, masks_t, token_ids_t)
    print('End warmup')
    
    begin = time.time()
    optimize_output = model(images_t, masks_t, token_ids_t)
    end = time.time()
    
    print(f'Inference time: {end - begin:.3f}s')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
    
    torch.testing.assert_close(output, optimize_output)
    
    
if __name__ == '__main__':
    main()