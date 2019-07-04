import sys
from skimage import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as pre
import scipy.io as sp

fileName = 'ensemble2'
signal = pd.read_csv('./CSVs/' + fileName + '.csv', header=None)

signalX = signal[0]
signalY = signal[1]
print(signalX)


plt.plot(signalX,signalY)
plt.scatter(signalX,signalY)
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (arb units)')
plt.show()
