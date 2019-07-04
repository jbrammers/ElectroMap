import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics

fileName = '294427'
df = pd.read_csv('./CSVs/' + fileName + 'Drop.csv', index_col=0).transpose().to_numpy()
lf = pd.read_csv('./CSVs/' + fileName + 'Labels.csv', index_col=0).transpose().to_numpy()

scores = []
X_train, X_test, y_train, y_test = tts(df, np.ravel(lf), test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, pred))
