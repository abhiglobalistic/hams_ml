from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
import numpy as np

#local imports
from Utility import get_models
from Graphs import Graphs

class CrossValidation:

    def __init__(self,X,y,raw_data):

        self.X = X
        self.y = y
        self.raw_data = raw_data


    def compute_CV(self):


        for name, clf in get_models().items():

            scores = cross_val_score(clf, self.X, self.y, cv=10)

            print('Model Evaluation : {0},{1}'.format(name, (scores.mean(), scores.std() * 2)))

            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            plots = Graphs(self.X, self.y, self.raw_data)

            plots.plot_validationCurve(title=name + ' Validation Curve',scores=scores)


