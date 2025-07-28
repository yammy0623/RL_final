import torch
print(torch.cuda.is_available())  # 是否支持 CUDA
print(torch.cuda.device_count())  # 可用的 GPU 數量