import numpy as np
from sklearn.metrics import classification_report

#local imports
from Utility import get_raw_data, get_Xy, get_train_test, get_models
from Graphs import Graphs


class Normalize:


    def __init__(self,X,y,raw_data,cols_to_scale=['col3','col64','col294']):

        self.X = X
        self.y = y
        self.data = raw_data
        self.cols_to_scale = cols_to_scale


    def simple_scaling(self):

        print('Simple scaling ...')

        for col in self.cols_to_scale:
            self.data[col] = self.data[col] / self.data[col].max()

        return self.data


    def min_max_scaling(self):

        print('Min max scaling...')

        for col in self.cols_to_scale:
            self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())

        return self.data


    def z_score_scaling(self):

        print('Z_score scaling ...')

        for col in self.cols_to_scale:
            self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[col].std()

        return self.data


    def compute_Scaling(self):

        print('Scaling features... ')

        scaling_methods = {
            "simple_scaling": self.simple_scaling,
            "min_max_scaling": self.min_max_scaling,
            "z_score_scaling": self.z_score_scaling
        }

        target_names = np.array(['A', 'B', 'C', 'D', 'E'])

        for name,method in scaling_methods.items():

            data = method()
            X = data.drop('target',axis=1)
            y = data['target']

            X_train, X_test, y_train, y_test = get_train_test(features=X, target=y, test_size=0.25)

            print('Scaling method : {0}'.format(name))
            for name, clf in get_models().items():
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                clf_report = classification_report(y_test, predictions,
                                                   target_names=target_names)
                score = clf.score(X_test, y_test)

                print('Model Evaluation : {0}, Accuracy : {1}'.format(name,score))
                print('Report: {}'.format(clf_report))

                plots = Graphs(self.X, self.y, self.data)

                plots.plot_ConfusionMatrix(y_test=y_test, y_pred=predictions, classes=target_names,
                                           title=name + ' Normaliztion')
