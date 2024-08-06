import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
import pydicom
import numpy as np
import os
from PIL import Image

# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.5) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Cifar_CNN(nn.Module):
    def __init__(self):
        super(Cifar_CNN, self).__init__()
       
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(64,64 , 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5,384)
        self.dropout = nn.Dropout(p=0.5) 
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        
    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten image input
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
        

class mnist_CNN(nn.Module):
    def __init__(self):
        super(mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x





# NLST DATALOADER
class NLSTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Walk through all directories and subdirectories
        for root, _, files in os.walk(root_dir):
            # Determine label based on the directory path containing 'CC' or 'NC'
            if 'CC' in root:
                label = 1  # Cancer
            elif 'NC' in root:
                label = 0  # No cancer
            else:
                continue

            for file_name in files:
                if file_name.endswith('.png'):
                    self.data.append(os.path.join(root, file_name))
                    self.labels.append(label)
                    # Print statement for debugging purposes
                    # print(f"Found PNG file: {os.path.join(root, file_name)} with label: {label}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        png_path = self.data[idx]
        label = self.labels[idx]

        # Open the image file
        image = Image.open(png_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

class NLSTDataset_dicom(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Walk through all directories and subdirectories
        for root, _, files in os.walk(root_dir):
            # Determine label based on the directory path containing 'CC' or 'NC'
            if 'CC' in root:
                label = 1  # Cancer
            elif 'NC' in root:
                label = 0  # No cancer
            else:
                continue
            
            for file_name in files:
                if file_name.endswith('.dcm'):
                    self.data.append(os.path.join(root, file_name))
                    self.labels.append(label)
                    # Print statement for debugging purposes
                    # print(f"Found DICOM file: {os.path.join(root, file_name)} with label: {label}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dcm_path = self.data[idx]
        label = self.labels[idx]
        
        dcm = pydicom.dcmread(dcm_path)
        image = dcm.pixel_array

        image = image.astype(np.float32) / np.max(image)

        # Replicate the single channel to create a 3-channel image # uncomment out for resnet
        image = np.stack([image] * 3, axis=-1)
        
        # exit()
        if self.transform:
            image = self.transform(image)
        
        return image, label



class ResNetLoader:
    def __init__(self, model_name='resnet18', pretrained=True, num_classes=1000):
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = self._load_model()

    def _load_model(self):
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        else:
            raise ValueError("Invalid model name. Choose either 'resnet18' or 'resnet50'.")

        if self.num_classes != 1000:
            # Modify the final fully connected layer to match the number of classes
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)

        return model

    def get_model(self):
        return self.model

class ResNetModelHandler:
    def __init__(self, model_name='resnet18', pretrained=True, num_classes=1000):
        self.loader = ResNetLoader(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        self.model = self.loader.get_model()

    def get_model(self):
        return self.model



class CustomResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(CustomResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # If you want to change the number of output classes
        if num_classes != 1000:
            # Replace the last fully connected layer
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

class NLST_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(NLST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Ensure output is (64, 8, 8)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class NLST_CNN___(nn.Module):
    def __init__(self, num_classes):
        super(NLST_CNN___, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Ensure output is (64, 8, 8)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x