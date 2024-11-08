import threading
import torch
import tensorrt as trt

from functools import reduce

try:
    from cuda import cudart
except ImportError:
    logger.warning(
        'please pip3 install cuda-python.')
    
class TrtExecutor:
    def __init__(self, engine_file_path, device_id=0):
        err_type = cudart.cudaSetDevice(device_id)
        if err_type[0] != cudart.cudaError_t.cudaSuccess:
           raise RuntimeError(f'set cuda device in trt failed, reason: {err_type}')
        self.device_id = device_id
       
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')
        with open(engine_file_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.io_info = self.get_io_info()
        self.context = self.engine.create_execution_context()
        outputs = self._get_outputs_info()
        self.outputs = self.allocate_buffers(outputs)
        
        self.lock = threading.Lock()
        
    def __del__(self):
        self.engine = None
        self.context = None
        for o in self.outputs:
            cudart.cudaFree(o)
        self.outputs = None
    
    def get_io_info(self):
        def to_torch_dtype(trt_dtype):
            tb = {
                trt.DataType.BOOL: torch.bool,
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT32: torch.int32,
                trt.DataType.INT8: torch.int8,
            }
            return tb[trt_dtype]
        io_info = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            io_info.append([name, is_input, shape, to_torch_dtype(dtype)])
        return io_info        
    
    def _get_datatype_size(self, dtype):
        return torch.tensor([], dtype=dtype).element_size()
    
    def _get_tensor_nbytes(self, tensor):
        return tensor.numel() * self._get_datatype_size(tensor.dtype)
    
    def _get_outputs_info(self):
        return [info for info in self.io_info if not info[1]]
    
    def execute(self, *input_data, stream_handle=None):
        with self.lock:
            if len(input_data) == 0:
                raise RuntimeError('input data are empty')
            if any([data.device.index != self.device_id for data in input_data]):
                raise RuntimeError('input data and trt executor are not on same device')
            
            bindings = []
            for data in input_data:
                bindings.append(data.data_ptr())
            bindings.extend(self.outputs)
            if stream_handle is None:
                stream_handle = torch.cuda.current_stream(self.device_id).cuda_stream

            self.context.execute_async_v2(bindings, stream_handle)
            
            outputs_info = self._get_outputs_info()
            output_tensor_list = []
            for info in outputs_info:
                output_tensor_list.append(torch.empty(size=info[2], dtype=info[3], device=f'cuda:{self.device_id}'))
            
            cudart.cudaDeviceSynchronize()
            
            for i in range(len(outputs_info)):
                cudart.cudaMemcpy(output_tensor_list[i].data_ptr(), self.outputs[i],
                                  self._get_tensor_nbytes(output_tensor_list[i]), 
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
            
            if len(output_tensor_list) == 1:
                return output_tensor_list[0]
            return output_tensor_list
    
    def allocate_buffers(self, io_info):
        return [cudart.cudaMalloc(reduce(lambda x, y: x * y, i[2]) * self._get_datatype_size(i[3]))[1] for i in io_info]
    
    def print_info(self):
        print("Batch dimension is", "implicit" if self.engine.has_implicit_batch_dimension else "explicit")
        for info in self.io_info:
            print("input" if info[1] else "output", info[0], info[2], info[3])

# import cv2
# import numpy as np
# import tensorrt as trt
# import pycuda.driver as cuda
# import torch
# from typing import Union, Optional, Sequence,Dict,Any
# import torchvision.transforms as transforms
# from PIL import Image
 
 
 
# def load_engine(self, engine_file_path):
#     TRT_LOGGER = trt.Logger()
#     assert os.path.exists(engine_file_path)
#     print("Reading engine from file {}".format(engine_file_path))
#     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#         return runtime.deserialize_cuda_engine(f.read())
    
 
# def engine_infer(self, engine, input_image):
#     """
#     engine: load_engine函数返回的trt模型引擎
#     input_image: 模型推理输入图像，尺寸为(batch_size, channel, height, width)
#     output：Unet模型推理的结果，尺寸为(batch_size, class_num, height, width)
#     """
#     batch_size = input_image.shape[0]
#     image_channel = input_image.shape[1]
#     image_height = input_image.shape[2]
#     image_width = input_image.shape[3]
 
#     with engine.create_execution_context() as context:
#         # Set input shape based on image dimensions for inference
#         context.set_binding_shape(engine.get_binding_index("input"), (batch_size, image_channel, image_height, image_width))
 
#         # Allocate host and device buffers
#         bindings = []
#         for binding in engine:
#             binding_idx = engine.get_binding_index(binding)
#             size = trt.volume(context.get_binding_shape(binding_idx))
#             dtype = trt.nptype(engine.get_binding_dtype(binding))
#             if engine.binding_is_input(binding):
#                 input_buffer = np.ascontiguousarray(input_image)
#                 input_memory = cuda.mem_alloc(input_image.nbytes)
#                 bindings.append(int(input_memory))
#             else:
#                 output_buffer = cuda.pagelocked_empty(size, dtype)
#                 output_memory = cuda.mem_alloc(output_buffer.nbytes)
#                 bindings.append(int(output_memory))
 
#         stream = cuda.Stream()
#         # Transfer input data to the GPU.
#         cuda.memcpy_htod_async(input_memory, input_buffer, stream)
#         # Run inference
#         context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#         # Transfer prediction output from the GPU.
#         cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
#         # Synchronize the stream
#         stream.synchronize()
 
#     output = np.reshape(output_buffer, (batch_size, self.num_classes, image_height, image_width))
#     return output