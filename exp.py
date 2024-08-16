from utils import wTest
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from utils import ResNetCustom
import sys

from argparse import ArgumentParser

pa = ArgumentParser()
pa.add_argument("exptFile", type=str, help="path to the exeriment h5 file")
pa.add_argument("predictor", type=str, choices=['reso', 'cent', 'one_over_reso'], help="resolution of beam-center")
pa.add_argument("modelPath", type=str, help="path to the model .net file")
pa.add_argument("--gpu", action="store_true", help="whether to use the GPU")
args = pa.parse_args()

if args.predictor == "cent":
    labs = ["cent_fast_train", "cent_slow_train"]
else:
    labs = ["reso"]
val_dataset = wTest(args.exptFile, labs)

val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False)

dev = "cpu"
if args.gpu:
    dev = "gpu"
net = ResNetCustom(num_classes=len(labs))
net.load_state_dict(torch.load(args.modelPath, map_location=torch.device(dev)))
net = net.to(dev)
net.eval()
val_loss = 0.0
criterion = torch.nn.MSELoss()
all_cent = []
all_res = []
with torch.no_grad():
    for val_imgs, val_labs in val_loader:
        val_imgs, val_labs = val_imgs.to(dev), val_labs.to(dev)
        outputs = net(val_imgs)
        loss = criterion(outputs, val_labs)
        val_loss += loss.mean().item()
        if args.predictor == "one_over_reso":
            outputs = 1 / outputs
        print(outputs)

        if args.predictor == "cent":
            x,y = outputs.numpy().ravel()
            all_cent.append((x,y))
        else:
            r, = outputs.numpy().ravel()
            all_res.append(r)
            
if args.predictor=="cent":
    print("Results:", np.mean(all_cent,0), np.std(all_cent,0))
else:
    print("Results:", np.mean(all_res), np.std(all_res))
val_loss /= len(val_loader)
print(val_loss)
