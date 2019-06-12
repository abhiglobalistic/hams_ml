from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
	"KNeighborsClassifier": KNeighborsClassifier(),
	"Naive_Bayes": GaussianNB(),
	"LogisticRegression": LogisticRegression(),
	"SVC": SVC(kernel="rbf", gamma="auto"),
	"DecisionTreeClassifier": DecisionTreeClassifier(),
	"RandomForestClassifier": RandomForestClassifier(n_estimators=100),
	"MLP": MLPClassifier()
}

def get_models(all=True,estimator_name='KNeighborsClassifier'):

    if all ==  False:

        return models[estimator_name]

    else:

        return models



