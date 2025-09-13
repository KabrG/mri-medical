import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import Food101
import torchvision.models as models
from PIL import Image

import os
import kagglehub

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Download latest version
path = kagglehub.dataset_download("ninadaithal/imagesoasis")
path = os.path.join(path, "Data")

# Dataset info: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
print("Path to dataset files:", path)

exit(0)
# 80% training, 20% testing

# Create a seed to randomly split the dataset
class CustomDataset(Dataset):
    # https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, dataset_path: str, isTrain: bool, target_transform=None, transform=None):
        # Get all the dataset locations
        non_path = os.path.join(dataset_path, "Non Demented")
        mild_path = os.path.join(dataset_path, "Mild Dementia")
        very_mild_path = os.path.join(dataset_path, "Very mild Dementia")
        moderate_path = os.path.join(dataset_path, "Moderate Dementia")

        # if isTrain:
        #     self.

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


# Create Dataset Classs


# Create a transform object for future use since we want to vary the incoming images
transforms = torch.nn.Sequential(


)






img = Image.open("/Users/kabirguron/.cache/kagglehub/datasets/ninadaithal/imagesoasis/versions/1/data/Mild Dementia/OAS1_0382_MR1_mpr-4_160.jpg")

print(img.size)
img.show()
# while(1):
#     pass
# First interperate the model







