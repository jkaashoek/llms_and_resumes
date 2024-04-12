from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class BinaryClassifier:
    def __init__(self, method='logistic_regression'):
        if method == 'logistic_regression':
            self.model = LogisticRegression()
        elif method == 'random_forest':
            self.model = RandomForestClassifier()
        elif method == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(100,))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)