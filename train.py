import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image

import os
import kagglehub

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Download latest version
dataset_path = kagglehub.dataset_download("ninadaithal/imagesoasis")
dataset_path = os.path.join(dataset_path, "Data")

# Dataset info: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
print("Path to dataset files:", dataset_path)

exit(0)
# 80% training, 20% testing

# Create a seed to randomly split the dataset
class CustomDataset(Dataset):
    # https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, dataset_path: str, isTrain: bool, seed: int, target_transform=None, transform=None):
        # Get all the dataset locations
        non_path = os.path.join(dataset_path, "Non Demented")
        mild_path = os.path.join(dataset_path, "Mild Dementia")
        very_mild_path = os.path.join(dataset_path, "Very mild Dementia")
        moderate_path = os.path.join(dataset_path, "Moderate Dementia")


    def __len__(self):
        pass

        return
    

    def __getitem__(self, index):

        
        pass
        return


# Create Dataset Class

# Create a transform object for future use since we want to vary the incoming images
transforms = torch.nn.Sequential(

)

img = Image.open("/Users/kabirguron/.cache/kagglehub/datasets/ninadaithal/imagesoasis/versions/1/data/Mild Dementia/OAS1_0382_MR1_mpr-4_160.jpg")



print(img.size)
img.show()


exit(0)


train_dataset = CustomDataset(dataset_path, True, 67)
test_dataset = CustomDataset(dataset_path, False, 67)


batch_size = 32

# Get the dataloaders
training_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batchsize, shuffle=False)


# Select the device. If CUDA is available (Nvidia GPU's), then it will use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Obtain the model (Try ResNet-50)
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)


# Check if a model aready exists
try:
    model.load_state_dict(torch.load("MRI_model.pth", map_location=device))
    print("Loaded existing weights.")
except FileNotFoundError:
    print("No existing weights found. Training from scratch.")


# Adjust learning rate for optimizer accordingly
# Adam optimizer ensures that each weight has its own learning weight
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function 
criterion = nn.CrossEntropyLoss()














