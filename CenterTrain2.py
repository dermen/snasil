#!/usr/bin/env python
# coding: utf-8

# In[7]:
#test comment

import torch
print(torch.cuda.is_available())
torch.cuda.is_available()
import h5py
import pylab as plt
print("test")
print("Found %d devices" %torch.cuda.device_count())


# In[8]:


import pylab as plt


# In[9]:


plt.plot(range(10))


# In[10]:


h = h5py.File("test_data/compressed12.h5", "r")


# In[11]:


i_img = 0


# In[12]:


labels = list(h['labels'].attrs['names'])


# In[13]:


labels


# In[14]:


ridx = labels.index("reso")


# In[15]:


r = h['labels'][i_img,ridx]


# In[16]:


xidx = labels.index("cent_fast_train")


# In[17]:


yidx = labels.index("cent_slow_train")


# In[18]:


x,y = h['labels'][i_img,[xidx, yidx]]


# In[19]:


plt.imshow(h['images'][i_img], vmax=20, cmap='gray_r')


# In[20]:


plt.plot( [x], [y], 'rx', ms=10)


# In[21]:


plt.title(f"res={r:.2f}, cent={x:.2f},{y:.2f}")


# In[22]:


plt.show()


# In[23]:


import os


# In[24]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import h5py
import torchvision.transforms as transforms
import numpy as py
import os
import pylab as plt
import argparse




# In[25]:


class arianaTest:
    
    def __init__(self, h5_file_path, label_name):
        self.file = h5py.File(h5_file_path, 'r')
        self.labels_names = list(self.file["labels"].attrs["names"])
    
    # Print available label names and their count
        print(self.labels_names, len(self.labels_names))
    
    # Check if provided label_name(s) are in the available labels
        if isinstance(label_name, list):
            for name in label_name:
                if name not in self.labels_names:
                    raise ValueError(f"Provided label_name '{name}' not found in the available labels: {self.labels_names}")
        else:
            if label_name not in self.labels_names:
                raise ValueError(f"Provided label_name '{label_name}' not found in the available labels: {self.labels_names}")
    
        self.labels = self.file['labels']
        self.images = self.file['images']
        self.label_name = label_name  # defined label_name here
    
        print(self.labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("out of range")
        img = self.images[index]
        img = img.reshape(1, 820, 820)  # Add channel dimension
        img = torch.tensor(img, dtype=torch.float32)  # Convert to tensor
        lab_index = self.get_labels_index(self.label_name) # find index
        assert isinstance(lab_index, list)
        lab = torch.tensor([self.labels[index, i] for i in lab_index], dtype=torch.float32)
        #else:
        #lab = torch.tensor(self.labels[index, lab_index], dtype=torch.float32)
        return img, lab
        
    def get_labels_index(self, label_name):
        if isinstance(label_name, list):
            return [self.labels_names.index(name) for name in label_name]
        else:
            return [self.labels_names.index(label_name)]
            
    def get_data_by_label(self, i_img, label_name):
        label_index = self.get_labels_index(label_name)
        return self.file['labels'][i_img, label_index][0]
        
    def get_xy_data(self, i_img, label_x, label_y):
        x_idx = self.get_labels_index(label_x)
        y_idx = self.get_labels_index(label_y)
        x = self.file['labels'][i_img, x_idx]
        y = self.file['labels'][i_img, y_idx]
        return x[0], y[0]
    def plot_image(self, i_img, label_x, label_y, label_r, vmax=20, cmap='gray_r'):
        r = self.get_data_by_label(i_img, label_r)
        x, y = self.get_xy_data(i_img, label_x, label_y)
        plt.imshow(self.file['images'][i_img], vmax=vmax, cmap=cmap)
        plt.plot(x, y, 'rx', ms = 10)
        plt.title(f"res={r:.2f}, cent={x:.2f},{y:.2f}")
        plt.show()
    


# In[26]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 202 * 202, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script for image dataset.')
    parser.add_argument('--train_h5', type=str, required=True, help='Path to the training HDF5 file.')
    parser.add_argument('--val_h5', type=str, required=True, help='Path to the validation HDF5 file.')
    parser.add_argument('--label_name', type=str, required=True, help='Name of the label.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    return parser.parse_args()



# In[27]:


file_path = "test_data/compressed12.h5"
label_name = ["beam_center_fast", "beam_center_slow"]
hdf5_dataset = arianaTest(file_path, label_name)
dataloader = DataLoader(hdf5_dataset, batch_size=10, shuffle=True)


# In[28]:


print(f"Number of labels: {len(hdf5_dataset)}")


# In[29]:


i_img = 151
label_name = "beam_center_fast"
data = hdf5_dataset.get_data_by_label(i_img, label_name)
print(f"Data for image {i_img} and label '{label_name}': {data}")


# In[30]:


label_x = "cent_fast_train"
label_y = "cent_slow_train"
x, y = hdf5_dataset.get_xy_data(i_img, label_x, label_y)
print(f"x data: {x}, y data: {y}")


# In[31]:


from torch.utils.data import DataLoader

reso_dataloader = DataLoader(hdf5_dataset, batch_size=64, shuffle=True)

label_r = "reso"
hdf5_dataset.plot_image(i_img, label_x, label_y, label_r)


# In[32]:


train_features, train_labels = next(iter(reso_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img ,cmap='gray_r', vmin=0, vmax=20)
plt.show()
print(f"Label: {label}")


# In[37]:


import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Args:
    train_h5 = file_path  
    val_h5 = file_path  
    label_name = ['cent_fast_train', 'cent_slow_train'] 
    epochs = 10
    batch_size = 10
    lr = 1e-4

args = Args()

# Print training settings
print(f"Training settings: {args}")

# Error handling for file paths
try:
    # Datasets and dataloaders
    print("Loading datasets")
    train_dataset = arianaTest(args.train_h5, args.label_name)
    val_dataset = arianaTest(args.val_h5, args.label_name)
    print("Datasets loaded successfully")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise
except ValueError as e:
    print(f"Value error: {e}")
    raise

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# loss function
net = Net()
criterion = nn.MSELoss(reduction = "none")
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-1)
verbose = False

def plot_losses(epochs, train_losses, val_losses):
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

print("Starting training")

train_losses = []
val_losses = []

for epoch in range(args.epochs):
    net.train()
    train_loss = 0.0
    
    for train_imgs, train_labs in train_loader:
        
        optimizer.zero_grad()
        outputs = net(train_imgs)
        loss_all = criterion(outputs, train_labs)
        loss = loss_all.mean()
        #from IPython import embed
        #embed()
        loss.backward()
        clip_grad_norm_(net.parameters(), 1e6)
        optimizer.step()
        loss_i = loss.item()
        train_loss += loss_i
        if verbose:
            print(loss_all)
    train_loss /= len(train_loader)

    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_imgs, val_labs in val_loader:
            outputs = net(val_imgs)
            loss = criterion(outputs, val_labs)
            val_loss += loss.mean().item()
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} done. train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

print("Training completed")

epochs = range(1, args.epochs + 1)
plot_losses(epochs, train_losses, val_losses)


# In[ ]:





# In[ ]:




