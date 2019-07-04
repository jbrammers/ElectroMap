import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

fileName = '294427'
df = pd.read_csv('./CSVs/' + fileName + 'Drop.csv', index_col=0).transpose().to_numpy()
lf = pd.read_csv('./CSVs/' + fileName + 'Labels.csv', index_col=0).transpose().to_numpy()

scores = []
X_train, X_test, y_train, y_test = tts(df, np.ravel(lf), test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


model = MLPClassifier(solver='adam', alpha=1e-1, power_t=1, activation='relu', max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(metrics.accuracy_score(pred, y_test))
