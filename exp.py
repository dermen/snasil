from utils import wTest
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from utils import ResNetCustom
import sys
import time

from argparse import ArgumentParser

start_time = time.time()

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
    dev = "cuda:0"
net = ResNetCustom(num_classes=len(labs))
net.load_state_dict(torch.load(args.modelPath, map_location=torch.device(dev)))
net = net.to(dev)
net.eval()
val_loss = 0.0
criterion = torch.nn.MSELoss()
all_cent = []
all_res = []
results = []
times = []
with torch.no_grad():
    for i_img, (val_imgs, val_labs) in enumerate(val_loader):
        t1 = time.time()
        val_imgs, val_labs = val_imgs.to(dev), val_labs.to(dev)
        outputs = net(val_imgs)
        t2 = time.time()
        times.append(t2 - t1)
        if args.predictor == "one_over_reso":
            outputs = 1 / outputs
        loss = criterion(outputs, val_labs)
        loss_i = loss.mean().item()
        val_loss += loss_i 
        print(f"Img {i_img} prediction:", outputs)

        if dev.startswith("cuda"):
            outputs = outputs.cpu()

        if args.predictor == "cent":
            x,y = outputs.numpy().ravel()
            all_cent.append((x,y))
        else:
            r, = outputs.numpy().ravel()
            all_res.append(r)
        results.append( (i_img, np.round(loss_i,5)))

    std_dev = np.std(times, ddof =1)
    median_val = np.median(times)

    print(f"Standard Deviation: {std_dev}")
    print(f"Median: {median_val}")

    

results = sorted(results, key=lambda x: x[1]) 
print("Lowest loss images:", results[:5])
print("Highest loss images:", results[-5:])

if args.predictor=="cent":
    print(f"Average {args.predictor} over images:", np.mean(all_cent,0), np.std(all_cent,0))
else:
    print(f"Average {args.predictor} over images:", np.mean(all_res), np.std(all_res))
val_loss /= len(val_loader)
print("Avergae MSE Loss for images:", val_loss)

end_time = time.time()

total_time = end_time - start_time
print(f"Total time: {total_time:.2f} seconds")

