from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict

class BinaryClassifier:
    def __init__(self, df, text_col, controls, y_col, method='logistic_regression'):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        self.data = Dataset.from_pandas(df)
        self.text_col = text_col
        self.controls = controls
        self.y_col = y_col

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
    
    def preprocess_function(self, examples):
        return self.tokenizer(examples[self.text_col], padding=True, truncation=True, return_tensors='pt')
    
    def prepare_data(self):
        return self.data.map(self.preprocess_function, batched = True)