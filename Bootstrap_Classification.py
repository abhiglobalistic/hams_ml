#local imports
from Data_Utility import get_raw_data, get_Xy, get_train_test
from models import get_models

#get raw data
data = get_raw_data(all_rows=True)

#get features and target
X,y,target_names = get_Xy(dataframe=data)

#get train and test
X_train,X_test, y_train, y_test = get_train_test(features=X,target=y,test_size=0.25)

print("Dataset description : {0}, {1}".format(X.shape,y.shape))


# Create pipeline for the process to check every model from the list
for name,clf in get_models().items():

    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    #clf_report = classification_report(y_test, predictions,
    #                            target_names=target_names.keys())
    score = clf.score(X_test,y_test)

    print('Model Evaluation : {0}, Accuracy : {1}'.format(name, score))
    #print('Report: {}'.format(clf_report))


