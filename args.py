import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(description='Training script for image dataset.')
    parser.add_argument('--train_h5', type=str, required=True, help='Path to the training HDF5 file.')
    parser.add_argument('--val_h5', type=str, required=True, help='Path to the validation HDF5 file.')
    parser.add_argument('--label_name', type=str, nargs = '+', required=True, help='Name of the label.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--model', type=str, choices= ['oldschool', 'resnet', 'efficientnet', 'maxvit'],default ='resnet', help='Choice of archtecture.')
    parser.add_argument('--optim', type=str, choices= ['Adam', 'SGD'], default = 'SGD',help='Optimizer')
    parser.add_argument('--save-dir', type=str, choices= ['/data/wgoh', '/data/aamiri'], help='Base directory to save models')
    parser.add_argument('--folder-name', type=str, required=True, help='Folder name within the save directory')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for optimizer')
    parser.add_argument('--dev', type=int, choices= [0,1], default =0, help='Device ID')
    parser.add_argument('--log-file', type=str, help='Path to the log file for plotting.')
    args = parser.parse_args()

    if not args.log_file:
        if not args.train_h5 or not args.val_h5 or not args.label_name or not args.folder_name:
            parser.error('--train_h5, --val_h5, --label_name, and --folder-name are required when not using --log-file')

    return args

