import scipy.io as sio
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import numpy as np
import csv, sys

# Load in mat file containing time vs fluorescence data
fileName = '294427'
matName = '/294427 0 nM[3000-16000_0(0)_ROI2].mat'
mapName = '/294427 LA ROI APD.txt'
mat = sio.loadmat('./Motion/' + fileName + matName)
dataMap = pd.read_csv('./Motion/' + fileName + mapName, header=None)

# Allocate one of the overlays to be the dataset with data in the format [x][y][time]
dataset = mat['aD_overlayP']
dict = {}
labels = {}
coords = []

# Assign each pixel to a dictionary entry
x = 0
y = 0
for i in dataset:
    for j in dataset[x]:
        coord = str(x) + "," + str(y)
        dict[coord] = dataset[x][y]
        coords.append(coord)
        y += 1
    y = 0
    x += 1


x = 0
y = 0
for i in dataMap:
    for j in dataMap[x]:
        coord = str(x) + "," + str(y)
        labels[coord] = [dataMap[x][y]]
        y += 1
    y = 0
    x += 1

# Converts to pandas dataframes, df is the data and lf is the labels
df = pd.DataFrame.from_dict(dict)
lf = pd.DataFrame.from_dict(labels)

# Removes all not a number entries from both dataframe and labels
# df_clean and lf_clean are the dataframes with all NaNs removed
df_clean = df.dropna(axis=1, how='all')

lf_clean = pd.DataFrame()
lf.fillna(0, axis=1)
for i in df:
    if df_clean.__contains__(i):
        if lf.get(i) is None:
            temp = {i: [0]}
            temp = pd.DataFrame.from_dict(temp)
            temp = temp.transpose()
            lf_clean = lf_clean.append(temp)
        else:
            lf_clean = lf_clean.append(lf.get(i))

lf_clean = lf_clean.transpose()

# Writes the dataframes to a csv for later use
df_clean.to_csv(('./CSVs/' + fileName + 'Drop.csv'))
lf_clean.to_csv(('./CSVs/' + fileName + 'Labels.csv'))

# Displays graphs for all pixels that have a signal shown
# for pixel in df_clean:
#     plt.plot(df_clean[pixel])
#         plt.title(pixel)
#         plt.ylabel('Ratio of intensity/time')
#         plt.xlabel('Frame')
#         plt.show()

# Presents chosen pixel in a scatter graph

pixel = df_clean.get('5,5')
x = pixel.index
y = pixel
plt.scatter(x, y)
plt.plot(y)
plt.ylabel('Intensity (A.U)')
plt.xlabel('Time (A.U)')
plt.show()
