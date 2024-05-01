import sys
sys.path.append('..')
from utils.classifier import BinaryClassifier
import pandas as pd 
import numpy as np

data = pd.read_csv('../resumes/profiles_with_updates.csv')
print(data.columns)
print(data.head())

classifier = BinaryClassifier(data, 'original', [], 'hired', method='logistic_regression')
print(classifier.model)
print(classifier.prepare_data())