import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from Data_Utility import get_raw_data, get_Xy, get_train_test
from scipy.stats import stats
data = get_raw_data()

x = data['target']
y = data['col294']

plt.bar(x,y)
plt.show()

