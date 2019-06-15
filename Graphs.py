import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


class Graphs:


    def __init__(self,X,y,raw_data):

        self.X = X
        self.y = y
        self.raw_data = raw_data

        np.set_printoptions(precision=2)



    def plot_ConfusionMatrix(self,y_test, y_pred, classes,normalize=True,title=None,cmap=plt.cm.Blues):

        self.title = 'Normalized confusion matrix'
        cm = confusion_matrix(y_test, y_pred)
        classes = classes[unique_labels(y_test, y_pred)]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        plt.show()


    def plot_ClassDistribution(self,data):

        A = data['target'].value_counts()[0]
        B = data['target'].value_counts()[1]
        C = data['target'].value_counts()[2]
        D = data['target'].value_counts()[3]
        E = data['target'].value_counts()[4]
        As_per = A / data.shape[0] * 100
        Bs_per = B / data.shape[0] * 100
        Cs_per = C / data.shape[0] * 100
        Ds_per = D / data.shape[0] * 100
        Es_per = E / data.shape[0] * 100

        plt.figure(figsize=(8, 6))
        sns.countplot(data['target'])

        plt.xlabel('Target')
        plt.xticks((0, 1, 2, 3, 4), ['Class A ({0:.2f}%)'.format(As_per),
                                     'Class B ({0:.2f}%)'.format(Bs_per),
                                     'Class C ({0:.2f}%)'.format(Cs_per),
                                     'Class D ({0:.2f}%)'.format(Ds_per),
                                     'Class E ({0:.2f}%)'.format(Es_per)])
        plt.ylabel('Count')
        plt.title('Training Set Target Distribution')

        plt.show()


    def plot_validationCurve(self,scores,title):


        plt.plot(scores)
        plt.ylabel('Accuracy')
        plt.title(title)

        plt.show()





