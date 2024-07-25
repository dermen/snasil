#!/usr/bin/env python
# coding: utf-8


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

h = h5py.File("/data/wgoh/test_data/compressed12.h5","r")

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


#Our first architecture.
""" class Net(nn.Module):
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
print(net) """

#Second Architecture (ResNet)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * 103 * 103, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


file_path = "/data/wgoh/test_data/compressed12.h5"
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

# From the first architecture.
""" criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net = Net()
net.train()
criterion = nn.MSELoss()  
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9) #optim.adder """

""" # Test the model with a sample tensor
sample_input = torch.randn(1, 1, 820, 820) 
output = net(sample_input) """


""" print("Output shape:", output.shape)
print("Output:", output)
 """

total_loss = 0.0

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script for image dataset.')
    parser.add_argument('--train_h5', type=str, required=True, help='Path to the training HDF5 file.')
    parser.add_argument('--val_h5', type=str, required=True, help='Path to the validation HDF5 file.')
    parser.add_argument('--label_name', type=str, required=True, help='Name of the label.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    return parser.parse_args()


class Args:
    train_h5 = 'train_master.hdf5' 
    val_h5 = 'compressed12.hdf5' 
    label_name = ['reso']
    epochs = 100
    batch_size = 10
    lr = 0.0001

args = Args()

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



def plot_losses(epochs, train_losses, val_losses):
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()


# From the first architecture.
""" # Model, loss function, optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.09)
verbose = False
print("Starting training by first Architecture")


from IPython import embed
#embed()
dev = "cuda:0"
net = net.to(dev)


#print("Using device", net.device)

save_frequency = 5
train_losses = []
val_losses = []

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
        model_path = f"model{epoch + 1}.net"
        torch.save(net.state_dict(), model_path)
        print(f"Model saved as {model_path}") """


#For the ResNetModel
net = ResNet18()
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.MSELoss()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(dev)
save_frequency = 5
train_losses = []
val_losses = []
print("Starting training by ResNet")

for epoch in range(args.epochs):
    net.train()
    train_loss = 0.0
    
    for train_imgs, train_labs in train_loader:
        train_imgs, train_labs = train_imgs.to(dev), train_labs.to(dev)
        optimizer.zero_grad()
        outputs = net(train_imgs)
        loss = criterion(outputs, train_labs)
        loss.backward()
        clip_grad_norm_(net.parameters(), 1e6)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_imgs, val_labs in val_loader:
            val_imgs, val_labs = val_imgs.to(dev), val_labs.to(dev)
            outputs = net(val_imgs)
            loss = criterion(outputs, val_labs)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} done. train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    if (epoch + 1) % save_frequency == 0:
        model_path = f"model{epoch + 1}.net"
        save_model(model_path, net)
        print(f"Model saved as {model_path}")

# Save the final model
save_model("FirstTrainFile", net)

# Compute and print losses for the validation images
results = compute_losses_hdf5(args.val_h5, net, "reso", batch_size=args.batch_size)
print(results)


# Saves the training losses and validation losses into a text file, 
# Two columns, one for training and second for validation losses.
with open("losses.txt", "w") as f:
    for t_loss, v_loss in zip(train_losses, val_losses):
        f.write(f"{t_loss}\t{v_loss}\n")

print("Training completed")

epochs = range(1, args.epochs + 1)
plot_losses(epochs, train_losses, val_losses)

save_model("FirstTrainFile", net)

# Load the saved model for computing losses on validation images.
model_path = "model100.net"
net = Net()
net.load_state_dict(torch.load(model_path))
net = net.to(dev)

# Compute losses on validation images.
results = compute_losses_hdf5(args.val_h5, net, "reso", batch_size=args.batch_size)
results_sorted = sorted(results, key=lambda x: x[1])

# Print the images with the highest and lowest losses.
print(f"Image with lowest loss: Index {results_sorted[0][0]}, Loss: {results_sorted[0][1]:.4f}")
print(f"Image with highest loss: Index {results_sorted[-1][0]}, Loss: {results_sorted[-1][1]:.4f}")

# Saves it to a file.
with open("sorted_validation_losses.txt", "w") as f:
    for idx, loss in results_sorted:
        f.write(f"Index: {idx}, Loss: {loss:.4f}\n")



#if args.load_model_name:
#    net = load_model(args.load_model_name, Net)

#results = compute_losses_hdf5(args.val_h5, net, args.label_name, args.batch_size)
#print(results)