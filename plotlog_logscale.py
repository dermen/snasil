import plotutils
import sys

logfile = sys.argv[1]
out = plotutils.parselog(logfile)
print(out)
epoch, train, val = zip(*out)
plotutils.plot_losses(epoch, train, val, log=True)
