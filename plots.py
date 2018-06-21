import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.axes import Axes
import time 
import sys
import matplotlib.cm as cm
import os
from os import listdir


def getExperimentsData(filename):
    f = open('logs/' + filename)
    text = f.read()
    i = 1
    parts = []
    done = False
    while not done:
        start = text.find('Experiment-'+str(i))
        end = text.find('Experiment-'+str(i+1))
        i += 1
        if end == -1:
            end = text.find('Mean')-1
            done = True
        parts.append(text[start:end-1])
    return parts
    
def getValues(parts, timing, i):
    if timing:
        Xs = re.findall('time: (\d*.\d*)', parts[i])
        plt.xlabel('Time elapsed (s)')
    else:
        Xs = re.findall('Iter-(\d*)', parts[i])
        plt.xlabel('Number of iterations')
    Ys = re.findall('validation: (\d*.\d*)', parts[i])
    plt.ylabel('Validation accuracy (%)')
    return np.array(Xs, dtype=float), np.array(Ys, dtype=float)

def plotDataFor(ax, filename, timing, notation):
    parts = getExperimentsData(filename)
    for i in range(0, len(parts)):
        Xs, Ys = getValues(parts, timing, i)
        ax.plot(Xs, Ys, color=notation, label= filename if i == 0 else None)

timing = False
if len(sys.argv) == 1:
    print('Provide log names to plot')
    exit(0)
else:
    if (sys.argv[1] == 't'):
        timing = True
        names = sys.argv[2:]
    else:
        names = sys.argv[1:]

if os.path.isdir('logs/'+ names[0]): 
    names = [names[0]+'/' + s for s in listdir('logs/'+names[0])]

legendnames = []
args = names.copy()
names = []
for arg in args:
    parts = arg.split('=')
    legendnames.append(parts[-1])
    names.append(parts[0])

plt.ion()
fig, ax = plt.subplots()
#fig.patch.set_facecolor('xkcd:ecru')
ax.set_facecolor('xkcd:pale blue')
exit = False
def handle_close(evt):
    global exit
    exit = True
fig.canvas.mpl_connect('close_event', handle_close)
colors = cm.rainbow(np.linspace(0, 1, len(names)))
while not exit:
    ax.cla()
    i = 0
    for name in names:
        plotDataFor(ax, name, timing, colors[i])
        i += 1
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, legendnames, loc=4)
    plt.draw()
    plt.pause(4)
