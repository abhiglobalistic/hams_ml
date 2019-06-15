import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier



target_names = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def mapping(dataframe):
    data = dataframe.replace({'target': target_names})
    return data

def get_raw_data(no_of_rows=10000,all_rows=False):
    # read the rows and assign column names for 296 columns with last column as target
    data = pd.read_csv('sample.csv', names=['col' + str(idx) for idx in range(1, 296)])
    data = data.rename(columns={'col295': 'target'})

    data = mapping(dataframe=data)

    if all_rows == True:
        return data
    else:
        return data[:no_of_rows]

def get_Xy(dataframe=None):

    if dataframe is None:
        raise AttributeError('Please provide a valid dataframe')


    X = dataframe.drop('target', axis=1)
    y = dataframe['target']


    return X,y,target_names


def get_train_test(features,target,test_size=0.20,random_state=152):

    X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)

    return X_train,X_test, y_train, y_test


def get_models(all=True,estimator_name='KNeighborsClassifier'):

    models = {
        "KNeighborsClassifier": KNeighborsClassifier(leaf_size=30,p=2),
        "Naive_Bayes": GaussianNB(),
        "LogisticRegression": LogisticRegression(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
        "MLP": MLPClassifier()
    }

    if all ==  False:

        return models[estimator_name]

    else:

        return models







