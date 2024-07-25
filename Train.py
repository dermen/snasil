#!/usr/bin/env python
# coding: utf-8


class Args:
    train_h5 = 'train_master.hdf5' 
    val_h5 = 'compressed12.hdf5' 
    label_name = ['reso']
    epochs = 100
    batch_size = 10
    lr = 0.0001

argsTest = Args()

from args import parse_arguments

args = parse_arguments()


""" def test_args(args1, args2):
    assert args1.train_h5 == args2.train_h5
    assert args1.val_h5 == args2.val_h5
    assert args1.label_name == args2.label_name
    assert args1.epochs == args2.epochs
    assert args1.batch_size == args2.batch_size
    assert args1.lr == args2.lr
    print('ok')
test_args(args, argsTest) """



import h5py
import pylab as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import h5py
import torchvision.transforms as transforms
import numpy as py
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from utils import save_model, load_model, compute_losses_hdf5

h = h5py.File(args.val_h5,"r")

i_img = 0

labels = list(h['labels'].attrs['names'])
labels
ridx = labels.index("reso")
r = h['labels'][i_img, ridx]
xidx = labels.index("cent_fast_train")
yidx = labels.index("cent_slow_train")
x,y = h['labels'][i_img,[xidx,yidx]]


plt.imshow(h['images'][i_img,[xidx,yidx]])
plt.plot([x],[y],'rx',ms=10)
plt.imshow(h['images'][i_img], vmax = 20, cmap='gray_r')
plt.plot([x],[y],'rx',ms=10)
plt.title(f"res={r:.2f}, cent={x:.2f},{y:.2f}")
plt.show()

print (torch.cuda.is_available())
print("Found %d devices" %torch.cuda.device_count())


i_img = 0
labels = list(h['labels'].attrs['names'])

class wTest:
    def __init__(self, h5_file_path, label_name):
        self.file = h5py.File(file_path, 'r')
        self.labels_names = list(self.file["labels"].attrs["names"])
        print (self.labels_names, len(self.labels_names))
        self.labels = self.file['labels']
        self.images = self.file['images']
        self.label_name = label_name # defined label_name here
        print (self.labels)
        
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 202 * 202, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

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

file_path = args.val_h5
label_name = "reso"
print("Label: " + label_name)
hdf5_dataset = wTest(file_path, label_name)
dataloader = DataLoader(hdf5_dataset, batch_size=10, shuffle=True)



label_r = "reso"
label_x = "cent_fast_train"
label_y = "cent_slow_train"
i_img = 0

reso_dataloader = DataLoader(hdf5_dataset, batch_size=64, shuffle=True)
label_r = "reso"

hdf5_dataset.plot_image(i_img, label_x, label_y, label_r)
train_features, train_labels = next(iter(reso_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img ,cmap='gray_r', vmin=0, vmax=20)
plt.show()
print(f"Label: {label}")




criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net = Net()
net.train()
criterion = nn.MSELoss()  
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9) #optim.adder

# Test the model with a sample tensor
sample_input = torch.randn(1, 1, 820, 820) 
output = net(sample_input)


print("Output shape:", output.shape)
print("Output:", output)


total_loss = 0.0

""" for i, data in enumerate(dataloader, 0):
    inputs, labels = data
    
    optimizer.zero_grad()
    outputs = net(inputs)

    #from IPython import embed
    #embed()
    
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    print(f'Batch {i+1}, Loss: {loss.item()}')
numBatches = len(dataloader)
total_loss = total_loss / numBatches

print('Finished Training %d' % numBatches)
print('Total loss after 1 epoch:', total_loss)

sample_input = torch.randn(1, 1, 820, 820)
output = net(sample_input)
print("Output shape:", output.shape)
print("Output:", output)
 """






print(f"Training settings: {args}")

# Error handling for file paths
try:
    # Datasets and dataloaders
    print("Loading datasets")
    train_dataset = wTest(args.train_h5, args.label_name)
    val_dataset = wTest(args.val_h5, args.label_name)
    print("Datasets loaded successfully")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise
except ValueError as e:
    print(f"Value error: {e}")
    raise

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Model, loss function, optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.09)
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

from IPython import embed
#embed()
dev = "cuda:0"
net = net.to(dev)


#print("Using device", net.device)

save_frequency = 5

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


    print(f"Epoch {epoch+1} done. train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    if (epoch + 1) % save_frequency == 0:
        # Load the model100 into the utils load_model to find which validation iamges have the highest and lowest loss. 
        # Save the training losses and validation losses into a text file, two columns, one for training and second for validation losses.
        model_path = f"model{epoch + 1}.net"
        torch.save(net.state_dict(), model_path)
        print(f"Model saved as {model_path}")

print("Training completed")

epochs = range(1, args.epochs + 1)
plot_losses(epochs, train_losses, val_losses)

#save_model("FirstTrainFile", net)
#if args.load_model_name:
#    net = load_model(args.load_model_name, Net)

#results = compute_losses_hdf5(args.val_h5, net, args.label_name, args.batch_size)
#print(results)