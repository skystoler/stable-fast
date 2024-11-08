import torch
import time
import torch.nn.functional as F
import os
from sfast.utils.image_utils import load_image, pil_to_numpy, numpy_to_pt
from sfast.compilers.ocr_compiler import CompilationConfig, compile_print_reco_model
from sfast.utils.trt import TrtExecutor
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

SOURCE_DIR = "/root/autodl-tmp/data/"
IMAGE_DIR = SOURCE_DIR + "image/"
MODEL_DIR = SOURCE_DIR + "model/"
TRT_PATH = MODEL_DIR + 'prt_ft13_emptyrecog_invert_0411_147_opt.ts'
PRINT_RECO_PATH = MODEL_DIR + 'uniocr_wfeat_241024152028_uniocr_add_spzonly_241023_85.pth'
PRINT_FONT_PATH = MODEL_DIR + 'font47_240724_ft47_a100_mlp6240711_230.pth'
PRINT_CHAR_SEG_PATH = MODEL_DIR + 'loc_std2_ft13_atv6_0305_376.pth'
HW_RECO_PATH = MODEL_DIR + 'hw_832_blank_xingcao_1211_ep203.pth'
PRINT_IMG_URL = IMAGE_DIR + '924_SPLIT_72311_40986bc2-48a1-11ef-a600-00163e153e23.jpg'
HW_IMG_URL = IMAGE_DIR + 'e2bca13b366942768b722310d8aeee27.jpeg'
TRT_SO_PATH = SOURCE_DIR + 'trt_executor_extension_py38.so'
DATA_LIST_PATH = SOURCE_DIR + 'download_188_20241106072127.txt'
TRT_MODEL_PATH = MODEL_DIR + ''

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

def prof(func):
    start = torch.cuda.Event(enable_timing=True)  # Create a start event
    end = torch.cuda.Event(enable_timing=True)  # Create an end event
    start.record()
    func()
    start = torch.cuda.Event(enable_timing=True)  # Create a start event
    end = torch.cuda.Event(enable_timing=True)  # Create an end event
    start.record()


def test_single_pic():
    print(torch.__version__)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(PRINT_RECO_PATH, map_location=device)
    model.eval()
    
    images_t, masks_t, token_ids_t = prepare_input(img_url=PRINT_IMG_URL, device=device)
    """
    not compile
    """
    print('Begin warmup')
    for _ in range(5):
        model(images_t, masks_t, token_ids_t)
    print('End warmup')
    
    start = torch.cuda.Event(enable_timing=True)  # Create a start event
    end = torch.cuda.Event(enable_timing=True)  # Create an end event
    start.record()
    output = model(images_t, masks_t, token_ids_t)
    start = torch.cuda.Event(enable_timing=True)  # Create a start event
    end = torch.cuda.Event(enable_timing=True)  # Create an end event
    end.record()
    torch.cuda.synchronize()
    infer_time = start.elapsed_time(end)
    print(f'Orignal Inference time: {infer_time:.3f}s')
    # peak_mem = torch.cuda.max_memory_allocated()
    # print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
    
    """
    compile
    """
    begin = time.time()
    compiled_model = torch.compile(model, mode='max-autotune')
    end = time.time()
    print(f'Compile time: {end - begin:.3f}s')
    
    # config = CompilationConfig.Default()
    # try:
    #     import triton
    #     config.enable_triton = True
    # except ImportError:
    #     print('Triton not installed, skip')
        
    # compile_print_reco_model(model, config)

    print('Begin warmup')
    for _ in range(5):
        compiled_model(images_t, masks_t, token_ids_t)
    print('End warmup')
    
    start = torch.cuda.Event(enable_timing=True)  # Create a start event
    end = torch.cuda.Event(enable_timing=True)  # Create an end event
    start.record()
    optimize_output = model(images_t, masks_t, token_ids_t)
    start = torch.cuda.Event(enable_timing=True)  # Create a start event
    end = torch.cuda.Event(enable_timing=True)  # Create an end event
    end.record()
    torch.cuda.synchronize()
    compiled_infer_time = start.elapsed_time(end)
    print(f'Compiled Inference time: {compiled_infer_time:.3f}s')
    # peak_mem = torch.cuda.max_memory_allocated()
    # print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
    
    speed_up = infer_time / compiled_infer_time
    print(f'Speed up: {speed_up:.3f}')
    
    torch.testing.assert_close(output, optimize_output)
        
    
def test_compile_print_reco_model():
    print(torch.__version__)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(PRINT_RECO_PATH, map_location=device)
    model.eval()
    
    images_t, masks_t, token_ids_t = prepare_input(img_url=PRINT_IMG_URL, device=device)
    """
    not compile
    """
    print('Begin warmup')
    for _ in range(5):
        model(images_t, masks_t, token_ids_t)
    print('End warmup')
    
    """
    compile
    """
    begin = time.time()
    compiled_model = torch.compile(model, mode ="max-autotune")
    end = time.time()
    print(f'Compile time: {end - begin:.3f}s')
    
    """
    self-compile
    """
    # config = CompilationConfig.Default()
    # try:
    #     import triton
    #     config.enable_triton = True
    # except ImportError:
    #     print('Triton not installed, skip')
    # begin = time.time() 
    # compiled_model = compile_print_reco_model(model, config)
    # end = time.time()
    # print(f'Compile time: {end - begin:.3f}s')
    
    print('Begin warmup')
    for _ in range(5):
        compiled_model(images_t, masks_t, token_ids_t)
    print('End warmup')
    
    speed_up_sum = 0.0
    test_cnt = 50
    images = os.listdir(IMAGE_DIR)
    for index, img in enumerate(images[:test_cnt]):
        print(index + 1, img)
        img_url = IMAGE_DIR + img
        images_t, masks_t, token_ids_t = prepare_input(img_url=img_url, device=device)

        begin = time.time()
        output = model(images_t, masks_t, token_ids_t)
        end = time.time()
        infer_time = end - begin 
        print(f'Orignal Inference time: {infer_time:.3f}s')
        # peak_mem = torch.cuda.max_memory_allocated()
        # print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
        
        begin = time.time()
        optimize_output = compiled_model(images_t, masks_t, token_ids_t)
        end = time.time()
        compiled_infer_time = end - begin
        print(f'Compiled Inference time: {compiled_infer_time:.3f}s')
        # peak_mem = torch.cuda.max_memory_allocated()
        # print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
        
        speed_up = infer_time / compiled_infer_time
        print(f'Speed up: {speed_up:.3f}')
        speed_up_sum += speed_up
        
        #torch.testing.assert_close(output, optimize_output)
    print(f'Avergae Speed up: {(speed_up_sum / test_cnt):.3f}')


def test_trt():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    executor = TrtExecutor(engine_file_path=TRT_MODEL_PATH, device_id=device)
    
    images_t, masks_t, token_ids_t = prepare_input(img_url=PRINT_IMG_URL, device=device)
    
    # print('Begin warmup')
    # for _ in range(5):
    #     pass
    # print('End warmup')
    
    begin = time.time()
    output = executor.execute(images_t, masks_t, token_ids_t)
    end = time.time()
    infer_time = end - begin
    print(f'TRT Inference time: {infer_time:.3f}s')


def test_model_bottleneck():
    print(torch.__version__)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(PRINT_RECO_PATH, map_location=device)
    model.eval()
    
    images_t, masks_t, token_ids_t = prepare_input(img_url=PRINT_IMG_URL, device=device)
    
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            model(images_t, masks_t, token_ids_t)
            
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
if __name__ == '__main__':
    test_single_pic()
    # test_compile_print_reco_model()
    # test_trt()
    # test_model_bottleneck()