import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
import matplotlib.pyplot as plt

fileName = '294427'
df = pd.read_csv('./CSVs/' + fileName + 'Drop.csv', index_col=0).transpose().to_numpy()
lf = pd.read_csv('./CSVs/' + fileName + 'Labels.csv', index_col=0).transpose().to_numpy()

scores = []
X_train, X_test, y_train, y_test = tts(df, np.ravel(lf), test_size=0.2)

# Below for loop used to determine the best number of neighbours

# rangeK = range(1, 25)
# for k in rangeK:
#     model = knn(n_neighbors=k)
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     scores.append(metrics.accuracy_score(pred, y_test))
#
# plt.plot(rangeK, scores)
# plt.show()

model = knn(n_neighbors=1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(metrics.accuracy_score(pred, y_test))
