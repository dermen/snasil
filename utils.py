import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pylab as plt
import math
import copy
from typing import Sequence, Union, Callable, Optional, List
from torch import Tensor
from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig, Conv2dNormActivation

def save_model(model_name, model):
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")

def load_model(model_name, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_name))
    model.eval()
    print(f"Model loaded from {model_name}")
    return model

class wTest:
    def __init__(self, h5_file_path, label_name):
        self.file = h5py.File(h5_file_path, 'r')
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
        import pylab as plt
        plt.imshow(self.file['images'][i_img], vmax=vmax, cmap=cmap)

        plt.plot(x, y, 'rx', ms = 10)
        plt.title(f"res={r:.2f}, cent={x:.2f},{y:.2f}")
        plt.show()

""" class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, label_name):
        self.hdf5_file = hdf5_file
        self.label_name = label_name
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f[label_name])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            img = torch.tensor(f['images'][idx], dtype=torch.float32)
            label = torch.tensor(f[self.label_name][idx], dtype=torch.float32)
        return img, label
         """
def compute_losses_hdf5(hdf5_file, model, label_name, batch_size=10):
   

    dataset = wTest(hdf5_file, label_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    results = []

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        outputs = model(imgs)
        losses = criterion(outputs, labels)
        for i in range(len(imgs)):
            index = batch_idx * batch_size + i
            loss = losses[i].item()
            results.append((index, loss))

    results.sort(key=lambda x: x[1])
    return results


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



class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, (MBConvConfig, FusedMBConvConfig)) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[Union[MBConvConfig, FusedMBConvConfig]]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                1, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def efficientnet_b0_config() -> Sequence[Union[MBConvConfig, FusedMBConvConfig]]:
    return [
        MBConvConfig(1, 3, 32, 16, 1, 1, 0.25),
        MBConvConfig(6, 3, 16, 24, 2, 2, 0.25),
        MBConvConfig(6, 5, 24, 40, 2, 2, 0.25),
        MBConvConfig(6, 3, 40, 80, 2, 2, 0.25),
        MBConvConfig(6, 5, 80, 112, 1, 1, 0.25),
        MBConvConfig(6, 5, 112, 192, 2, 2, 0.25),
        MBConvConfig(6, 3, 192, 320, 1, 1, 0.25),
    ]