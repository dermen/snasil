import pylab as plt

def plot_losses(epochs, train_losses, val_losses, log=False):
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    if log:
        plt.gca().set_yscale("log")
    plt.show()

def parselog(logfile):
    f=open(logfile, "r")
    lines = f.readlines()
    data = []
    for l in lines:
        if not l.startswith("Epoch"):
            continue
        ls = l.split()
        epoch = int(ls[1])
        trainloss = float(ls[4].replace(",", ""))
        valloss = float(ls[6].replace("\n", ""))

        data.append((epoch, trainloss, valloss))
    
    return data
