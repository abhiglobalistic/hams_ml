from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report
import numpy as np

#local imports
from Utility import get_train_test, get_models
from Graphs import Graphs


class FeatureEngineering:

    def __init__(self,X,y,raw_data):

        self.X = X
        self.y = y
        self.raw_data = raw_data


    def varianceThreshold(self):

        print('Variance Threshold... ')
        print('Intial feature count : {0}'.format(self.X.shape[1]))

        # reduce the no of columns based on variance
        selections = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_new = selections.fit_transform(self.X)

        return X_new


    def selectKbest(self):

        print('select k=20 best... ')
        X_new = SelectKBest(score_func=f_classif,k=20).fit_transform(self.X,self.y)

        return X_new


    def compute_Features(self):


        print('Computing features...  ')

        feature_sel_methods = {

            "selectKbest": self.selectKbest,
            "varianceThreshold": self.varianceThreshold

        }


        for name,method in feature_sel_methods.items():

            X_new = method()

            print("Dataset no of features after " + name + " selection and target size {0}, {1}".format(X_new.shape, self.y.shape))

            X_train, X_test, y_train, y_test = get_train_test(features=X_new, target=self.y, test_size=0.30)

            # Create a pipeline for the process to check every model from the list

            target_names = np.array(['A', 'B', 'C', 'D', 'E'])

            for name, clf in get_models().items():

                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                clf_report = classification_report(y_test, predictions,
                                                   target_names=target_names)
                score = clf.score(X_test, y_test)

                print('Model Evaluation : {0},{1}'.format(name,score))
                print('Report: {}'.format(clf_report))

                plots = Graphs(self.X, self.y, self.raw_data)

                plots.plot_ConfusionMatrix(y_test=y_test, y_pred=predictions, classes=target_names,
                                           title=name + ' Feature Engineering')

