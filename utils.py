import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def save_model(model_name, model):
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")

def load_model(model_name, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_name))
    model.eval()
    print(f"Model loaded from {model_name}")
    return model

class HDF5Dataset(Dataset):
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
        
def compute_losses_hdf5(hdf5_file, model, label_name, batch_size=10):
   

    dataset = HDF5Dataset(hdf5_file, label_name)
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
