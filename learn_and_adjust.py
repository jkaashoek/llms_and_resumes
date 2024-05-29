import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import edsl
from edsl.questions import QuestionFreeText, QuestionYesNo
from edsl import Agent, Model, Survey
from sklearn.metrics import confusion_matrix
import argparse

argparser = argparse.ArgumentParser(description='Learn and adjust based on data')
argparser.add_argument('data_path', help='Path to decision file')
argparser.add_argument('--text_column', default='text', help='Name of the column containing text')

args = argparser.parse_args()

# Check to see the data path exists
data_path = args.data_path

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f'{data_path} does not exist')

# Get the original text as well as the decision
columns = data.columns
key_col = columns[1] # The second column is the one we'll join on 

# The original file with the text
orig_stem = Path(data_path).stem
orig_fp = orig_stem[:orig_stem.rfind('_')]
orig_data = pd.read_csv(f'data/sampled_profiles/{orig_fp}.csv')
orig_data[key_col] = orig_data[key_col].astype(str)
data[key_col] = data[key_col].astype(str)

if args.text_column not in orig_data.columns:
    raise ValueError(f'{args.text_column} not in columns')

orig_data = orig_data[[key_col, args.text_column]]
data_with_decision = pd.merge(orig_data, data, on=key_col)


# We want to keep some of the profiles separate
# I want the same profiles across run. Just take the first 80%
train_profiles = data_with_decision[key_col].values[:int(len(data_with_decision) * 0.8)]
# train_profiles = data_with_decision[key_col].sample(frac=0.8)
train_data = data_with_decision[data_with_decision[key_col].isin(train_profiles)]
test_data = data_with_decision[~data_with_decision[key_col].isin(train_profiles)]

# Set up the history of decisions for each manager
managers = data['persona'].unique()
print("Mangers", managers)
history_strs = []
for m in managers:
    hist = ''
    m_data = train_data[train_data['persona'] == m]
    for i, row in m_data.iterrows():
        hist += row[args.text_column] + f"\nDecision: {row['decision']}" + '\n------\n'
    history_strs.append(hist)

# We'll perform three actions on the data: see if we can learn the decision, adjust the text with a generic improvement, and adjust the text with a tailored improvement
learn_persona = "You will be given a set of candidate profiles and hiring decisions from a hiring manager. Your goal is to understand how the hiring manager makes their decisions. You will be asked to predict whether the hiring manager will hire the person or not based on the text provided."
generic_persona = "You are an expert hiring manager. You will be presented with a profile of an individual and will be asked to improve the profile so they are more likely to be hired. "
adjust_persona = "You will be given a set of candidate profiles and hiring decisions from a hiring manager. Your goal is to understand how the hiring manager makes their decisions. You will then be asked to adjust new profiles so that they are more likely to be hired by the manager." 

# We need to do this for each managers TODO: this doesn't work yet. We don't need to run the generic agent. The improvement should be the same. 
learn_agents = [Agent(name = f'{managers[i]}_learner', traits = {'persona': learn_persona + '\n\n' + h}) for i, h in enumerate(history_strs)]
generic_agent = [Agent(name = f'generic', traits = {'persona': generic_persona})]
adjust_agents = [Agent(name = f'{managers[i]}_adjuster', traits = {'persona': adjust_persona + '\n\n' + h}) for i, h in enumerate(history_strs)]

# Our question lists 
learn_qs = [
    QuestionYesNo(question_name = f'q_{test_data[key_col].values[i]}',
                  question_text = f'Would you hire this person? \n\n{test_data[args.text_column].values[i]}') 
                  for i in range(len(test_data))
]
generic_adjust = [
    QuestionFreeText(question_name = f'q_{test_data[key_col].values[i]}', 
                     question_text = f'Improve the profile so the candidate is more likely to be hired. If there are no improvements to be made, return the original text. \n\n{test_data[args.text_column].values[i]}') 
                     for i in range(len(test_data))
]
adjust_qs = [
    QuestionFreeText(question_name = f'q_{test_data[key_col].values[i]}', 
                     question_text = f'Adjust this profile to make it more likely to be hired given previous decision. Include only the new profile as your response. If there are no improvements to be made, return the original text. \n\n{test_data[args.text_column].values[i]}') 
                     for i in range(len(test_data))
]

# Set up surveys
model = Model('gpt-4-1106-preview')

def clean_esdl(edsl_res : edsl.results, search_str : str = r'_(.*)'):
    '''
    Function to clean the results from the EDSL
    '''
    df = edsl_res.select("agent.*", "answer.*").to_pandas().melt()
    df = df[df['variable'].str.contains('answer.q_')]
    pattern = re.compile(search_str)
    df[key_col] = df['variable'].apply(lambda x: pattern.search(x).group(1) if pattern.search(x) else '')
    # df['decision'] = np.where(df['value'] == 'Yes', 1, 0)
    df = df.drop(columns = ['variable'])
    df = df[[key_col, 'value']]
    return df

# Learning
print("\n----Running learning survey----")
learn_res = Survey(learn_qs).by(learn_agents).by(model).run()

# Did the learner learn anything?
learn_df = clean_esdl(learn_res)
learn_df['decision'] = np.where(learn_df['value'] == 'Yes', 1, 0)
learn_df = learn_df.drop(columns = ['value']) 
test_with_pred = pd.merge(test_data, learn_df, on=key_col, suffixes=('_true', '_pred'))
print("Learning results: confusion matrix")
print(confusion_matrix(test_with_pred['decision_true'], test_with_pred['decision_pred']))

# Generic adjustment
print("\n----Running generic adjustment survey----")
generic_res = Survey(generic_adjust).by(generic_agent).by(model).run()
generic_df = clean_esdl(generic_res, search_str=r'_(.*)')
print(generic_df.head())
generic_df.to_csv(f'computed_objects/{orig_fp}_generic_adjustment.csv')

# Tailored adjustment
print("\n----Running tailored adjustment survey----")
adjust_res = Survey(adjust_qs).by(adjust_agents).by(model).run()
adjust_df = clean_esdl(adjust_res, search_str=r'_(.*)')
adjust_df.to_csv(f'computed_objects/{orig_fp}_tailored_adjustment.csv')
print(adjust_df)

