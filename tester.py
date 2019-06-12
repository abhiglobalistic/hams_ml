import pandas as pd
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv('sample.csv',names = ['col'+str(idx) for idx in range(1,296)])
data = data.rename(columns={'col295': 'target'})
#data = data[:10000]
print(data.describe())

X = data.drop('target',axis=1)


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = sel.fit_transform(X)

print(X.shape)