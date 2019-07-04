import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem

fileName = '294427'
df = pd.read_csv('./CSVs/' + fileName + 'Drop.csv', index_col=0).transpose().to_numpy()
lf = pd.read_csv('./CSVs/' + fileName + 'Labels.csv', index_col=0).transpose().to_numpy()

scores = []
X_train, X_test, y_train, y_test = tts(df, np.ravel(lf), test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# pca = PCA(n_components=1)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)

# nystoem = Nystroem(gamma=.2, random_state=1)
# X_train = nystoem.fit_transform(X_train)
# X_test = nystoem.fit_transform(X_test)

model = SVC(C=10, kernel='rbf')
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(metrics.accuracy_score(pred, y_test))
#
# model = SGDClassifier()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(metrics.accuracy_score(pred, y_test))
#
# model = LinearSVC()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print(metrics.accuracy_score(pred, y_test))
