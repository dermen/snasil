import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(description='Training script for image dataset.')
    parser.add_argument('--train_h5', type=str, required=True, help='Path to the training HDF5 file.')
    parser.add_argument('--val_h5', type=str, required=True, help='Path to the validation HDF5 file.')
    parser.add_argument('--label_name', type=str, nargs = '+', required=True, help='Name of the label.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--model', type=str, choices= ['oldschool', 'resnet'],default ='resnet', help='Choice of archtecture.')
    parser.add_argument('momentum', type=float, default=0.0, help='Momentum for optimizer')
    args = parser.parse_args()
    return args

