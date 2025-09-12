import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import Food101
import torchvision.models as models
from PIL import Image
import kagglehub

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Download latest version
path = kagglehub.dataset_download("ninadaithal/imagesoasis")

# Dataset info: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
print("Path to dataset files:", path)


# Create a transform object for future use since we want to vary the incoming images
transforms = torch.nn.Sequential(



)






img = Image.open("/Users/kabirguron/.cache/kagglehub/datasets/ninadaithal/imagesoasis/versions/1/data/Mild Dementia/OAS1_0382_MR1_mpr-4_160.jpg")

print(img.size)
img.show()
# while(1):
#     pass
# First interperate the model







