from PIL import Image
import torch
from torchvision import models, transforms

print("✅ Python is running & packages loaded")

# ❗ No download needed:
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

model.eval()
print("✅ Model created (no pretrained weights)")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

img = Image.new("RGB", (224,224), color="brown")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(x)

print("✅ Forward pass OK, output shape:", out.shape)

