import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

target_names = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

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










