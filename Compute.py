import warnings
warnings.filterwarnings("ignore")


# -------Local imports--------#
import Bootstrap_Classification
import Normalizing
import FeatureEngineering
import Class_Sampling
import Cross_Validation

from Utility import get_raw_data,get_Xy,get_train_test,get_models

raw_data = get_raw_data(all_rows=True)
X,y,target_names = get_Xy(raw_data)


def performBootstrap():

    bsClf = Bootstrap_Classification.BootstrapClassification(X,y,raw_data)

    bsClf.compute_Bootstrap()

def performNormalizing():

    norm = Normalizing.Normalize(X,y,raw_data)

    norm.compute_Scaling()

def performFeatureEngg():

    feat = FeatureEngineering.FeatureEngineering(X,y,raw_data)

    feat.compute_Features()


def performClassSampling():

    clsSamp = Class_Sampling.Sampling(X,y,raw_data)

    clsSamp.compute_all_Sampling()


def performCrossValidation():

    crsVal = Cross_Validation.CrossValidation(X,y,raw_data)

    crsVal.compute_CV()




if __name__ == '__main__':

    pipeline = {
        'Bootstrap':performBootstrap,
        'Normalizing':performNormalizing,
        'FeatureEngg':performFeatureEngg,
        'ClassSampling':performClassSampling,
        'CrossValidation':performCrossValidation
    }

    for name,method in pipeline.items():

        method()


