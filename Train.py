#!/usr/bin/env python
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
from utils import wTest, Net, EfficientNet, _efficientnet, _efficientnet_conf, MaxVit
from torch.optim import Adam
import re


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

#if args.log_file:
#    print(f"Reading log file: {args.log_file}")
#
#    with open(args.log_file, 'r') as file:
#        log_data = file.read()
#
#    print("Log data read successfully.")
#
#    pattern = r"Epoch (\d+) done\. train_loss: ([\d\.]+), val_loss: ([\d\.]+)"
#    epochs = []
#    train_losses = []
#    val_losses = []
#
#    matches = list(re.finditer(pattern, log_data))
#    print(f"Found {len(matches)} matches.")
#
#    for match in re.finditer(pattern, log_data):
#        epochs.append(int(match.group(1)))
#        train_losses.append(float(match.group(2)))
#        val_losses.append(float(match.group(3)))
#
#    if epochs:
#
#        print(f"Epochs: {epochs}")
#        print(f"Train Losses: {train_losses}")
#        print(f"Validation Losses: {val_losses}")
#
#        # Plot the data
#        plt.figure(figsize=(10, 6))
#        plt.plot(epochs, train_losses, label='Train Loss')
#        plt.plot(epochs, val_losses, label='Validation Loss')
#        plt.xlabel('Epoch')
#        plt.ylabel('Loss')
#        plt.title('Training and Validation Loss Over Epochs')
#        plt.legend()
#        plt.grid(True)
#        plt.show()
#    else:
#        print("No matching log data found.")
#
#    exit()
#
#else:
#    h = h5py.File(args.val_h5,"r")

print (torch.cuda.is_available())
print("Found %d devices" %torch.cuda.device_count())

from utils import ResNetCustom

total_loss = 0.0

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

#0=a


print('Training images', len(train_dataset))
print('Val images', len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Model, loss function, optimizer
if args.model == 'resnet':
    net = ResNetCustom(num_classes=len(args.label_name))
elif args.model == 'efficientnet':
    inverted_residual_setting, last_channel = _efficientnet_conf('efficientnet_b0', width_mult=1.0, depth_mult=1.0)
    net = _efficientnet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
        weights=None,
        progress=True,
        num_classes=len(args.label_name)
    )
elif args.model == 'maxvit':
    net = MaxVit()
else:
    net = Net()


#add adam optimizer

criterion = nn.MSELoss()
if args.optim == "Adam": 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum)
verbose = False
print("Using Optimizer", optimizer)

from plotutils import plot_losses

print("Starting training")

train_losses = []
val_losses = []

#from IPython import embed
#embed()
dev = "cuda:%d" %args.dev
net = net.to(dev)


print("Using device", dev)

full_save_dir = os.path.join('.', args.folder_name)
os.makedirs(full_save_dir, exist_ok=True)

save_frequency = 5
from tqdm import tqdm
for epoch in range(args.epochs):
    net.train()
    train_loss = 0.0
    
    for train_imgs, train_labs in tqdm(train_loader):
        train_imgs, train_labs = train_imgs.to(dev), train_labs.to(dev)
        optimizer.zero_grad()
        outputs = net(train_imgs)
        loss_all = criterion(outputs, train_labs)
        loss = criterion(outputs, train_labs)
        loss.backward()
        clip_grad_norm_(net.parameters(), 1e6)
        optimizer.step()

        #from IPython import embed
        #embed()
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
        model_path = os.path.join(full_save_dir, f"model{epoch + 1}.net")
        torch.save(net.state_dict(), model_path)
        print(f"Model saved as {model_path}")

with open("losses.txt", "w") as f:
    for t_loss, v_loss in zip(train_losses, val_losses):
        f.write(f"{t_loss}\t{v_loss}\n")

print("Training completed")

epochs = range(1, args.epochs + 1)
plot_losses(epochs, train_losses, val_losses)



#save_model("FirstTrainFile", net)
#if args.load_model_name:
#    net = load_model(args.load_model_name, Net)

#results = compute_losses_hdf5(args.val_h5, net, args.label_name, args.batch_size)
#print(results)


