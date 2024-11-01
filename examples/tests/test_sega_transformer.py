import time
import torch
import torch.nn.functional as F
import types
from sfast.utils.image_utils import load_image, pil_to_numpy, numpy_to_pt

SOURCE_DIR = "/root/autodl-tmp/data/"
TRT_PATH = SOURCE_DIR + 'prt_ft13_emptyrecog_invert_0411_147_opt.ts'
PATH = SOURCE_DIR + 'uniocr_wfeat_240919134829_uniocr_ft47xkat6_240903_75.pth'
IMG_URL = SOURCE_DIR + 'text_rectification.jpg'

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

"""
forward() will use a special optimized implementation described in
`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
conditions are met:

- Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
    argument ``requires_grad``
- training is disabled (using ``.eval()``)
- batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
- activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
- at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
- if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
    nor ``src_key_padding_mask`` is passed
- the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
    unless the caller has manually modified one without modifying the other)
"""

from sageattention import sageattn

# def replace_attn(module, att_mask):
#     module_output = module
#     if isinstance(module, torch.nn.modules.MultiheadAttention):
#         # qkv = module.qkv
#         # dim = qkv.weight.shape[1] * module.num_heads
        
#         module_output = SageAttention(dim, module.num_heads, attn_mask=att_mask)
#     for name, child in module.named_children():
#         module_output.add_module(name, replace_attn(child, att_mask))
#     del module
#     return module_output
    

def main():
    print(torch.__version__)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(PATH, map_location=device)
    model.eval()
    images_t, masks_t, token_ids_t = prepare_input(img_url=IMG_URL, device=device)
    
    """
    not compile
    """
    begin = time.time()
    output = model(images_t, masks_t, token_ids_t)
    end = time.time()
    
    print(f'Inference time: {end - begin:.3f}s')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
    
    """
    compile
    """
    
    for (name, children) in model.base_architecture.transformer_encoder.layers.named_children():
        for (n, c) in children.named_children():
            if c.original_name == "MultiheadAttention":
                pass
                # torch._C._jit_pass_inline(c.graph)
                # sfast._C._jit_graph_constant_node_with_fucntion(c.graph)
                # for node in c.graph.findAllNodes("prim::CallFunction"):
                #     qkv_inputs = node.inputs()[:-1]
                #     print(qkv_inputs)
                # for node in c.graph.findAllNodes("prim::Constant"):
                #     node_type = node.output().type()
                #     if not isinstance(node_type, torch.BoolType) and not isinstance(node_type, torch.NoneType):
                        
                #         # torch._C.tryToGraphFunction(node)
                #         # replaceFirstUseWith()
                        
                #         setattr(c, node, sageattn)
                        
    model = torch.compile(model, mode='max-autotune')
    
    # replace_attn(model)
    begin = time.time()
    optimize_output = model(images_t, masks_t, token_ids_t)
    end = time.time()
    
    print(f'Inference time: {end - begin:.3f}s')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')

    torch.testing.assert_close(output, optimize_output)
    
if __name__ == '__main__':
    main()