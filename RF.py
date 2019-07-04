from sklearn.ensemble import RandomForestRegressor as rf
import pandas as pd

fileName = '294427Drop'
signal = pd.read_csv('./CSVs/' + fileName + '.csv')
#
# print(signal.head())
# print(signal.describe())
# print(signal.shape)
