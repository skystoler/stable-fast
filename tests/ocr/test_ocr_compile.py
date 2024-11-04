import torch
import time
import torch.nn.functional as F

from sfast.utils.image_utils import load_image, pil_to_numpy, numpy_to_pt
from sfast.compilers.ocr_compiler import CompilationConfig, compile_print_reco_model

SOURCE_DIR = "/root/autodl-tmp/data/"
TRT_PATH = SOURCE_DIR + 'prt_ft13_emptyrecog_invert_0411_147_opt.ts'
PRINT_RECO_PATH = SOURCE_DIR + 'uniocr_wfeat_240919134829_uniocr_ft47xkat6_240903_75.pth'
PRINT_FONT_PATH = SOURCE_DIR + 'font47_240724_ft47_a100_mlp6240711_230.pth'
PRINT_CHAR_SEG_PATH = SOURCE_DIR + 'loc_std2_ft13_atv6_0305_376.pth'
IMG_URL = SOURCE_DIR + 'text_rectification.jpg'
TRT_SO_PATH = SOURCE_DIR + 'trt_executor_extension_py38.so'

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
    
def test_print_reco_model():
    print(torch.__version__)
    # torch.classes.load_library(TRT_SO_PATH)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(PRINT_RECO_PATH, map_location=device)
    model.eval()
    
    images_t, masks_t, token_ids_t = prepare_input(img_url=IMG_URL, device=device)
    """
    not compile
    """
    print('Begin warmup')
    for _ in range(3):
        model(images_t, masks_t, token_ids_t)
    print('End warmup')
    
    begin = time.time()
    output = model(images_t, masks_t, token_ids_t)
    end = time.time()
    
    print(f'Inference time: {end - begin:.3f}s')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
    
    """
    compile
    """
    model = torch.compile(model, mode='max-autotune')
    
    # config = CompilationConfig.Default()
    # try:
    #     import triton
    #     config.enable_triton = True
    # except ImportError:
    #     print('Triton not installed, skip')
        
    # compile_print_reco_model(model, config)

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
    test_print_reco_model()