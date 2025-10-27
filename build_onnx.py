import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

IM_SIZE = 512
device = "cuda"

weights = DeepLabV3_ResNet50_Weights.DEFAULT
base = deeplabv3_resnet50(weights = weights).to(device)

class DeeplabOut(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]

model = DeeplabOut(base).to(device)

dummy = torch.randn(1, 3, IM_SIZE, IM_SIZE, device=device)
torch.onnx.export(
        model, dummy, "deeplabv3_mnv3_512.onnx",
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes=None)

print("OK")
