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
import random
import re

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

r_seed = 42
random.seed(r_seed) # Random seed used for shuffling

# Download latest version
dataset_path = kagglehub.dataset_download("ninadaithal/imagesoasis")
dataset_path = os.path.join(dataset_path, "Data")

# Dataset info: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
print("Path to dataset files:", dataset_path)


# 80% training, 20% testing



# dataset_path = os.path.join(dataset_path, "Very mild Dementia")
# OAS1_0186_MR1_mpr-3_132.jpg


# Create Dataset Class

# Create a transform object for future use since we want to vary the incoming images
i_transforms = transforms.Compose([
    transforms.Grayscale(1),
    # transforms.CenterCrop((228, 228)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # default mean and median
])


# Create a seed to randomly split the dataset
class CustomDataset(Dataset):

    def rand_file_list(self, file_path: str, people_num_list: list)->list:
        # Assume that the dataset incoming is shuffled 
        folder_file_paths = []

        # Regex pattern that will be reused
        pattern = re.compile(r"OAS1_(\d{4})", re.IGNORECASE)

        # Loop through the incoming names and append file paths
        for person in people_num_list:

            for file in os.scandir(file_path):
                
                # Check if it is a file
                if file.is_file():

                    # Get the name of the file
                    m = pattern.search(file.name)

                    if m and m.group(1) == person: # If it matches person str
                        folder_file_paths.append(file.path)
                    else:
                        pass # Move on to the next guy
                    

        # Shuffle before sending out
        random.shuffle(folder_file_paths)

        return folder_file_paths
    
    @staticmethod
    def unique_patient_ids(full_dir: str, isTrain: bool, seed: int = 42):
        """
        Example 'OAS1_0023_MR1_mpr-3_127.jpg' -> patient id 23
        """
        patient_ids = set()
        pattern = re.compile(r"OAS1_(\d{4})", re.IGNORECASE)

        # full_dir = os.path.join(dataset_path, demention_dir)
        print("Full dir:", full_dir)
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


    # https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, dataset_path: str, isTrain: bool, seed: int, img_transforms: transforms, target_transform=None):
        print(dataset_path)
        self.img_transforms = img_transforms
        dementia_types = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
        self.entire_file_list = []
        self.label_list = []
        

        # non = 0, very mild = 1, mild = 2, moderate = 3
        for label, type in enumerate(dementia_types):
            
            full_dir = os.path.join(dataset_path, type)

            # Generate people list
            people_list = self.unique_patient_ids(full_dir, isTrain, seed)
            # print("People list:", people_list)
            
            # Get all the files
            temp_arr = self.rand_file_list(full_dir, people_list)

            # Append to big file list
            self.entire_file_list += temp_arr

            # Add to label list
            self.label_list += [label for x in temp_arr]



    def __len__(self):
        
        return len(self.entire_file_list)

    

    def __getitem__(self, index):
        path = self.entire_file_list[index] 
        img = Image.open(path)
        img = self.img_transforms(img)

        return img, self.label_list[index]


train_dataset = CustomDataset(dataset_path, True, 42, i_transforms)
print(train_dataset.__len__())
print(train_dataset.__getitem__(train_dataset.__len__() -1))

exit()



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


train_dataset = CustomDataset(dataset_path, True, 42)
test_dataset = CustomDataset(dataset_path, False, 42)


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




