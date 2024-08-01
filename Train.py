#!/usr/bin/env python
# coding: utf-8




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
from utils import wTest, Net, EfficientNet, _efficientnet, _efficientnet_conf, MaxVit
from torch.optim import Adam


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


from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from typing import Type, Union, List, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetCustom(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNetCustom, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



# Create a model instance
net = ResNetCustom()

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
net = ResNetCustom()
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

#0=a


print('Training images', len(train_dataset))
print('Val images', len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Model, loss function, optimizer
if args.model == 'resnet':
    net = ResNetCustom()
elif args.model == 'efficientnet':
    inverted_residual_setting, last_channel = _efficientnet_conf('efficientnet_b0', width_mult=1.0, depth_mult=1.0)
    net = _efficientnet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
        weights=None,
        progress=True,
        num_classes=1
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

full_save_dir = os.path.join(args.save_dir, args.folder_name)
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

# Saves the training losses and validation losses into a text file, 
# Two columns, one for training and second for validation losses.
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