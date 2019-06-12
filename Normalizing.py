import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold

#local imports
from Data_Utility import get_raw_data, get_Xy, get_train_test
from models import get_models


data = None
cols_to_scale = ['col3','col64','col294']

def set_data():
    global data
    data = get_raw_data()

def simple_scaling():

    for col in cols_to_scale:
        data[col] = data[col] / data[col].max()

    return data



def min_max_scaling():

    for col in cols_to_scale:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    return data

def z_score_scaling():

    for col in cols_to_scale:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data



set_data()

scaling_methods = [simple_scaling,min_max_scaling,z_score_scaling]


scaling_methods = {
	"simple_scaling": simple_scaling,
	"min_max_scaling": min_max_scaling,
	"z_score_scaling": z_score_scaling
}

for name,method in scaling_methods.items():

    X, y, target_names = get_Xy(method())

    #combine variance with normalization, reduce the no of columns based on variance
    # comment this section to only used normaliztion with all features
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X = sel.fit_transform(X)

    X_train, X_test, y_train, y_test = get_train_test(features=X, target=y, test_size=0.25)

    print('Scaling method : {0}'.format(name))
    for name, clf in get_models().items():
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        clf_report = classification_report(y_test, predictions,
                                           target_names=target_names.keys())
        score = clf.score(X_test, y_test)

        print('Model Evaluation : {0}, Accuracy : {1}'.format(name,score))
        #print('Report: {}'.format(clf_report))
