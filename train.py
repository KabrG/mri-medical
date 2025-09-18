import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.models as models
# from PIL import Image
from PIL import ImageDraw, ImageFont, Image


import os, re, random
import kagglehub
import random
import re

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

r_seed = 93
# random.seed(r_seed) # Random seed used for shuffling

# Download latest version
dataset_path = kagglehub.dataset_download("ninadaithal/imagesoasis")
dataset_path = os.path.join(dataset_path, "Data")

# Dataset info: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
print("Path to dataset files:", dataset_path)

# Create Dataset Class

# Create a transform object for future use since we want to vary the incoming images
i_transforms = transforms.Compose([

    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   

])

test_transforms = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   

])



# Create a seed to randomly split the dataset
class CustomDataset(Dataset):

    @staticmethod
    def rand_file_list(file_path: str, people_num_list: list)->list:

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
        self.img_transforms = img_transforms
        dementia_types = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
        self.entire_file_list = []
        self.label_list = []
        

        # non = 0, very mild = 1, mild = 2, moderate = 3
        for label, type in enumerate(dementia_types):
            if label == 0:
                continue # Come back to this
            
            full_dir = os.path.join(dataset_path, type)

            # Generate people list
            people_list = self.unique_patient_ids(full_dir, isTrain, seed)
            
            # Get all the files
            temp_arr = self.rand_file_list(full_dir, people_list)
            # print(f"Label {label} has {len(temp_arr)} images")

            # Append to big file list
            self.entire_file_list += temp_arr

            # Let's try binary classification only
            y = label
            if label != 0:
                y = 1
                
            # Add to label list
            self.label_list += [y for x in temp_arr]

        # Now, add label 0. Needs to appear as many times as other class
        full_dir = os.path.join(dataset_path, "Non Demented")

        # Generate people list
        people_list = self.unique_patient_ids(full_dir, isTrain, seed)
        
        # Get all the files
        temp_arr = self.rand_file_list(full_dir, people_list)

        # Shuffle
        random.shuffle(temp_arr)

        # Remove elements until they are equal
        while len(self.entire_file_list) < len(temp_arr):
            temp_arr.pop()

        # print("LOOK HERE", len(temp_arr), len(self.entire_file_list))
        # Add temp array to the full list and add the labels
        self.entire_file_list += temp_arr
        self.label_list += [0 for x in temp_arr]

        print(len(self.label_list), "and", len(self.entire_file_list))

    def __len__(self):
        
        return len(self.entire_file_list)

    
    def __getitem__(self, index):
        path = self.entire_file_list[index] 
        img = Image.open(path)
        img = self.img_transforms(img)

        return img, self.label_list[index]

class MRIModel(torch.nn.Module):
    # Resource: https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
    def __init__(self, freeze_backbone=True):
        super(MRIModel, self).__init__()

        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)

        # Modify first conv layer to accept 1 channel since Resnet50 expects 3 channels
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,      # grayscale
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

        # Initialize weights by averaging pretrained RGB channels
        with torch.no_grad():
            self.model.conv1.weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.model(x)




train_dataset = CustomDataset(dataset_path, True, r_seed, i_transforms)
test_dataset = CustomDataset(dataset_path, False, r_seed, test_transforms)

print("Created Dataset Objects")
print(train_dataset.__len__())
print(test_dataset.__getitem__(0))

# Only for displaying the image

# for i in range(10):
#     index = int(random.random()*len(train_dataset))
#     image, label = train_dataset[index]

#     # If it's a tensor, convert to PIL before displaying
#     to_pil = transforms.ToPILImage()
#     image = to_pil(image)

#     draw = ImageDraw.Draw(image)
#     draw.text((5, 10), "label: " + str(label), fill="white") 
#     draw.text((5, 20), "index: " + str(index), fill="white") 

#     image.show()

# exit()

batch_size = 32

# Get the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)


# Select the device. If CUDA is available (Nvidia GPU's), then it will use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = MRIModel(freeze_backbone=True).to(device)
# Check if a model aready exists
try:
    model.load_state_dict(torch.load("mri_model.pth", map_location=device))
    print("Loaded existing weights.")
except FileNotFoundError:
    print("No existing weights found. Training from scratch.")


# Adjust learning rate for optimizer accordingly
# Adam optimizer ensures that each weight has its own learning weight
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=(1e-4))


# Loss function
criterion = nn.CrossEntropyLoss()



def train():
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_dataloader.dataset)} "
                  f"({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}")
            
    
    running_loss += loss.item() * data.size(0)
    return running_loss / len(train_dataloader.dataset)


def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_dataloader:
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    accuracy = 100. * correct / len(test_dataloader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_dataloader.dataset)} "
          f"({accuracy:.2f}%)\n")

    return test_loss, accuracy



# Training and testing loop

print("Testing prior to training")

epochs = 20
test()
override = True
override_set = False

for epoch in range(1, epochs + 1): 
    train()
    test()

    # Unfreeze layer4 at epoch 5
    if epoch == 5 and not override:
        print("Unfreezing layer4...")
        for name, param in model.model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5   # lower LR for fine-tuning
        )

    # Unfreeze layer3 at epoch 10
    elif epoch == 10 and not override:
        print("Unfreezing layer3...")
        for name, param in model.model.named_parameters():
            if "layer3" in name:
                param.requires_grad = True
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-6
        )

    # Unfreeze layer2 at epoch 15
    elif epoch == 15 and not override:
        print("Unfreezing layer2...")
        for name, param in model.model.named_parameters():
            if "layer2" in name:
                param.requires_grad = True
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-6
        )

    elif override:
        print("Unfreezing all layers...")
        for name, param in model.model.named_parameters():
            if "layer2" in name:
                param.requires_grad = True
            elif "layer3" in name:
                param.requires_grad = True
            elif "layer4" in name:
                param.requires_grad = True

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-6
        )



    # Open file in read mode and read if it should continue running
    run_status = "stop" # By default, the program will stop
    try:
        with open("continue_running.txt", "r") as file:
            run_status = str(file.read()).lower()

    except Exception as p:
        print(p)

    if run_status == "stop":
        print("Halting training...")
        break
    else:
        print("Continue training")
        pass

torch.save(model.state_dict(), "mri_model.pth")
print("Model weights saved to mri_model.pth")

