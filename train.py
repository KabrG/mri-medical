import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image

import os, re, random
import kagglehub

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Download latest version
dataset_path = kagglehub.dataset_download("ninadaithal/imagesoasis")
dataset_path = os.path.join(dataset_path, "Data")

# Dataset info: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
print("Path to dataset files:", dataset_path)


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

        return 1


    def __getitem__(self, index):



        return 1, 1


def unique_patient_ids(dataset_path: str, demention_dir: str, isTrain: bool, seed: int = 42):
    """
    Example 'OAS1_0023_MR1_mpr-3_127.jpg' -> patient id 23
    """
    patient_ids = set()
    pattern = re.compile(r"OAS1_(\d{4})", re.IGNORECASE)

    full_dir = os.path.join(dataset_path, demention_dir)
    if not os.path.isdir(full_dir):
        return patient_ids
    for fname in os.listdir(full_dir):
        m = pattern.search(fname)
        if m:
            patient_ids.add(m.group(1))

    patient_ids = sorted(patient_ids)

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    split_index = int(0.8 * len(patient_ids))
    train_ids = patient_ids[:split_index]
    test_ids = patient_ids[split_index:]

    return train_ids if isTrain else test_ids


patients = unique_patient_ids(dataset_path, "Very mild Dementia", False)
print(f"Found {len(patients)} unique patients")
print(patients)

# Create Dataset Class

# Create a transform object for future use since we want to vary the incoming images
img_transforms = transforms.Compose([
    transforms.Grayscale(1),
    # transforms.CenterCrop((228, 228)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # default mean and median
])

temp_path = os.path.join(dataset_path, "Mild Dementia/OAS1_0382_MR1_mpr-4_160.jpg")

img = Image.open(temp_path)


print(img.size)

img = img_transforms(img)

print(img.size())


# Only for displaying the image
to_pil = transforms.ToPILImage()
img_pil = to_pil(img)

img_pil.show()

# Remember to remove exit
exit()




train_dataset = CustomDataset(dataset_path, True, 67)
test_dataset = CustomDataset(dataset_path, False, 67)


batch_size = 32

# Get the dataloaders
training_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)


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














