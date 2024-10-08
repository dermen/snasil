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
    
        print(self.labels_names, len(self.labels_names))
    
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

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output


# Create a model instance
net = Net()
print(net)
# Test with a single image tensor of shape (1, 1, 820, 820)
input_tensor_single = torch.randn(1, 1, 820, 820)
output_single = net(input_tensor_single)
print(output_single.shape)  # Should print torch.Size([1, 1])
# Test with a batch of 10 images tensor of shape (10, 1, 820, 820)
input_tensor_batch = torch.randn(10, 1, 820, 820)
output_batch = net(input_tensor_batch)
print(output_batch.shape)  # Should print torch.Size([10, 1])
file_path = “/data/aamiri/test_data/compressed12.h5”
label_name = “reso”
print(“Label: ” + label_name)
hdf5_dataset = arianaTest(file_path, label_name)
dataloader = DataLoader(hdf5_dataset, batch_size=10, shuffle=True)
label_r = “reso”
label_x = “cent_fast_train”
label_y = “cent_slow_train”
i_img = 0
reso_dataloader = DataLoader(hdf5_dataset, batch_size=64, shuffle=True)
label_r = “reso”
hdf5_dataset.plot_image(i_img, label_x, label_y, label_r)
train_features, train_labels = next(iter(reso_dataloader))
print(f”Feature batch shape: {train_features.size()}“)
print(f”Labels batch shape: {train_labels.size()}“)
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img ,cmap=‘gray_r’, vmin=0, vmax=20)
plt.show()
print(f”Label: {label}“)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net = Net()
net.train()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9) #optim.adder
# Test the model with a sample tensor
sample_input = torch.randn(1, 1, 820, 820)
output = net(sample_input)
print(“Output shape:“, output.shape)
print(“Output:“, output)
total_loss = 0.0

for i, data in enumerate(dataloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    #from IPython import embed
    #embed()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    print(f’Batch {i+1}, Loss: {loss.item()}’)
numBatches = len(dataloader)
total_loss = total_loss / numBatches
print(‘Finished Training %d’ % numBatches)
print(‘Total loss after 1 epoch:’, total_loss)
sample_input = torch.randn(1, 1, 820, 820)
output = net(sample_input)
print(“Output shape:“, output.shape)
print(“Output:“, output)
 
def parse_arguments():
    parser = argparse.ArgumentParser(description=‘Training script for image dataset.‘)
    parser.add_argument(‘--train_h5’, type=str, required=True, help=‘Path to the training HDF5 file.‘)
    parser.add_argument(‘--val_h5’, type=str, required=True, help=‘Path to the validation HDF5 file.‘)
    parser.add_argument(‘--label_name’, type=str, required=True, help=‘Name of the label.‘)
    parser.add_argument(‘--epochs’, type=int, default=10, help=‘Number of epochs to train.‘)
    parser.add_argument(‘--batch_size’, type=int, default=10, help=‘Batch size for training.‘)
    parser.add_argument(‘--lr’, type=float, default=0.001, help=‘Learning rate.’)
    return parser.parse_args()
class Args:
    train_h5 = ‘train_master.hdf5’
    val_h5 = ‘compressed12.hdf5’
    label_name = ['cent_fast_train', 'cent_slow_train']
    epochs = 100
    batch_size = 10
    lr = 0.0001
args = Args()
print(f”Training settings: {args}“)
# Error handling for file paths
try:
    # Datasets and dataloaders
    print(“Loading datasets”)
    train_dataset = arianaTest(args.train_h5, args.label_name)
    val_dataset = arianaTest(args.val_h5, args.label_name)
    print(“Datasets loaded successfully”)
except FileNotFoundError as e:
    print(f”File not found: {e}“)
    raise
except ValueError as e:
    print(f”Value error: {e}“)
    raise
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
# Model, loss function, optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.09)
verbose = False
def plot_losses(epochs, train_losses, val_losses):
    plt.plot(epochs, train_losses, label=‘Training Loss’)
    plt.plot(epochs, val_losses, label=‘Validation Loss’)
    plt.xlabel(‘Epochs’)
    plt.ylabel(‘Loss’)
    plt.title(‘Training and Validation Loss over Epochs’)
    plt.legend()
    plt.show()
print(“Starting training”)
train_losses = []
val_losses = []
from IPython import embed
#embed()
dev = “cuda:0"
net = net.to(dev)
#print(“Using device”, net.device)
for epoch in range(args.epochs):
    net.train()
    train_loss = 0.0
    for train_imgs, train_labs in train_loader:
        train_imgs, train_labs = train_imgs.to(dev), train_labs.to(dev)
        optimizer.zero_grad()
        outputs = net(train_imgs)
        loss_all = criterion(outputs, train_labs)
        loss = criterion(outputs, train_labs)
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
            val_imgs, val_labs = val_imgs.to(dev), val_labs.to(dev)
            outputs = net(val_imgs)
            loss = criterion(outputs, val_labs)
            val_loss += loss.mean().item()
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f”Epoch {epoch+1} done. train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}“)
print(“Training completed”)
epochs = range(1, args.epochs + 1)
plot_losses(epochs, train_losses, val_losses)
#save_model(“FirstTrainFile”, net)
#if args.load_model_name:
#    net = load_model(args.load_model_name, Net)
#results = compute_losses_hdf5(args.val_h5, net, args.label_name, args.batch_size)
#print(results)



