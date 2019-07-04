import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
import matplotlib.pyplot as plt


fileName = '294427'
df = pd.read_csv('./CSVs/' + fileName + 'Drop.csv', index_col=0).transpose().to_numpy()
lf = pd.read_csv('./CSVs/' + fileName + 'Labels.csv', index_col=0).transpose().to_numpy()

scores = []
X_train, X_test, y_train, y_test = tts(df, np.ravel(lf), test_size=0.2)


# Below for loop used to determine the best number of estimators
# rangeN = range(10, 500, 10)
# for n in rangeN:
#     avg = []
#     for k in range(1,5):
#         model = RandomForestClassifier(n_estimators=n, n_jobs=-1)
#         model.fit(X_train, y_train)
#         pred = model.predict(X_test)
#         avg.append(metrics.accuracy_score(y_test, pred))
#     scores.append(np.average(avg))
# plt.plot(rangeN, scores)
# plt.show()


model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, pred))
