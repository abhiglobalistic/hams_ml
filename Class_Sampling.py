from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

#local imports
from Utility import get_train_test, get_models
from Graphs import Graphs


class Sampling:

    def __init__(self,X,y,raw_data):

        self.X = X
        self.y = y
        self.raw_data = raw_data



        # Class count
        count_class_A, count_class_B, count_class_C, count_class_D, count_class_E = self.raw_data.target.value_counts()

        print('Class Count Intially... ')
        print()

        print('Class_A : {0}, Class_B : {1}, Class_C : {2}, Class_D : {3}, Class_E : {4}'.format(count_class_A,count_class_B,count_class_C,count_class_D,count_class_E))



    def perform_Over_SMOTE(self):
        print('Over sampling with SMOTE, adding synthetic data for minority classes')

        X_resampled, y_resampled = SMOTE().fit_resample(self.X, self.y)

        return X_resampled, y_resampled


    def perform_Under_ClusterCentroids(self):

        print('Under sampling with ClusterCentroids, preserves imformation')

        cc = ClusterCentroids(random_state=0)

        X_resampled, y_resampled = cc.fit_resample(self.X, self.y)

        return X_resampled, y_resampled


    def perform_Over_Random_Sampling(self):

        print('Random Over sampling...  ')

        ros = RandomOverSampler(random_state=0)

        X_resampled, y_resampled = ros.fit_resample(self.X, self.y)

        return X_resampled,y_resampled



    def perform_Under_Random_Sampling(self):

        print('Random Under sampling... ')

        rus = RandomUnderSampler(random_state=42)

        X_resampled, y_resampled = rus.fit_resample(self.X, self.y)

        return X_resampled, y_resampled




    def compute_all_Sampling(self):

        sampling_methods = {

            'Over_SMOTE': self.perform_Over_SMOTE,
            'Under_ClusterCentroids': self.perform_Under_ClusterCentroids,
            'Over Random Sampling': self.perform_Over_Random_Sampling,
            'Under Random Sampling': self.perform_Under_Random_Sampling
        }

        target_names = np.array(['A', 'B', 'C', 'D', 'E'])

        for name,method in sampling_methods.items():

            print('Running Sampling...')

            X_resampled, y_resampled = method()

            X_train, X_test, y_train, y_test = get_train_test(features=X_resampled, target=y_resampled, test_size=0.25)

            class_Counts = sorted(Counter(y_resampled).items())
            print('Method = {0}'.format(name), class_Counts)

            plots = Graphs(self.X, self.y, self.raw_data)

            dataF = pd.DataFrame(data=X_resampled,columns=['col' + str(idx) for idx in range(1, 295)])
            dataF['target'] = y_resampled

            print("After sampling : {0}".format(dataF.shape))

            plots.plot_ClassDistribution(data=dataF)

            print('Sampling method : {0}'.format(name))
            for name, clf in get_models().items():
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                clf_report = classification_report(y_test, predictions,
                                                   target_names=target_names)
                score = clf.score(X_test, y_test)

                print('Model Evaluation : {0}, Accuracy : {1}'.format(name, score))
                print('Report: {}'.format(clf_report))


                plots.plot_ConfusionMatrix(y_test=y_test, y_pred=predictions, classes=target_names,
                                           title=name + ' Class sampling')



