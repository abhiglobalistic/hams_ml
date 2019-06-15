from sklearn.metrics import classification_report
import numpy as np

#local imports
from Graphs import Graphs
from Utility import get_train_test,get_models


class BootstrapClassification:


    def __init__(self,X,y,raw_data):

        self.X = X
        self.y = y
        self.raw_data = raw_data


    def compute_Bootstrap(self):

        #get train and test
        X_train,X_test, y_train, y_test = get_train_test(features=self.X,target=self.y,test_size=0.30)


        print("Dataset description : {0}, {1}".format(self.X.shape,self.y.shape))

        target_names = np.array(['A','B','C','D','E'])

        # Create pipeline for the process to check every model from the list
        for name,clf in get_models().items():

            clf.fit(X_train,y_train)
            predictions = clf.predict(X_test)
            print(predictions)
            clf_report = classification_report(y_test, predictions,
                                       target_names=target_names)
            score = clf.score(X_test,y_test)

            print('Model Evaluation : {0}, Accuracy : {1}'.format(name, score))
            print('Report: {}'.format(clf_report))

            plots = Graphs(self.X,self.y,self.raw_data)

            plots.plot_ConfusionMatrix(y_test=y_test,y_pred=predictions,classes=target_names,title= name +' Bootstrap Classification')

