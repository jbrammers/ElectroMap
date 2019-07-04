import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fileName = '294427'
df = pd.read_csv('./CSVs/' + fileName + 'Drop.csv', index_col=0)


means = df.mean()

mean = np.average(means)
std_dev = np.std(means)
print(mean)
print(std_dev)
print(mean+std_dev*2)
print(mean-std_dev*2)

bounded_means = {}

for i in range(0,means.size-1):
    if mean+std_dev*2 < means[i] or mean-std_dev*2 > means[i]:
        pass
    else:
        bounded_means[i] = means[i]


df_std = pd.Series(bounded_means).to_frame()
df_mean = np.average(df)
print(df_mean)
print(df.head())
print(df.shape)
plt.scatter(y=df_std, x=df_std.index)
plt.show()
