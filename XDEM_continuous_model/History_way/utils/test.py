import torch
print("torch :", torch.__version__)
print("cuda? :", torch.cuda.is_available())
print("cuda runtime :", torch.version.cuda)
if torch.cuda.is_available():
    print("device :", torch.cuda.get_device_name(0))