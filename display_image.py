from utils import wTest
from argparse import ArgumentParser
pa = ArgumentParser()
pa.add_argument("h5", type=str, help="validation of experiment h5 file")
pa.add_argument("i", type=int, help="index from dataset")
pa.add_argument("--vmax", type=float, help="color scaler value (increase to lower brightness)")
args = pa.parse_args()
dataset = wTest(args.h5, "reso")
dataset.plot_image(args.i, "cent_fast_train", "cent_slow_train", "reso", vmax=args.vmax)
