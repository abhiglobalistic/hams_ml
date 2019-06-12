import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report

#local imports
from Data_Utility import get_raw_data, get_Xy, get_train_test
from models import get_models


#get raw data
data = get_raw_data()

#get train and test
X,y,target_names = get_Xy(dataframe=data)


#reduce the no of columns based on variance
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = sel.fit_transform(X)

varianceDF = pd.DataFrame(columns=['col'+str(i) for i in range(X.shape[1])],data=X)
varianceDF['target'] = y.values
varianceDF.to_csv('varianceDF.csv')

X_train,X_test, y_train, y_test = get_train_test(features=X,target=y,test_size=0.25)

#print("Dataset no of features after variance scaling and target size {0}, {1}".format(X.shape,y.shape))


# Create pipeline for the process to check every model from the list

for name,clf in get_models().items():

    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    clf_report = classification_report(y_test, predictions,
                                target_names=target_names.keys())
    score = clf.score(X_test,y_test)


    #print('Model Evaluation : {}'.format(name))
    #print('Report: {}'.format(clf_report))
    #print('Accuracy: {}'.format(score))
