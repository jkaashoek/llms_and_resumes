# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: edsl_env
#     language: python
#     name: python3
# ---

from datasets import load_dataset
from edsl import Model, Agent, Survey
from edsl.questions import QuestionFreeText
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

wnut = load_dataset("wnut_17")
label_list = wnut["train"].features[f"ner_tags"].feature.names
print(label_list)

# +
d0 = wnut["train"][0]
toks = d0["tokens"]
tags = d0["ner_tags"]

print(len(toks), len(tags))
print(toks, tags)
# -

example_sentence =  ' '.join(toks)
print(example_sentence)

print(Model.available())

# +
# models = [Model('gpt-4-1106-preview')]
models = [Model('gpt-3.5-turbo')]

# The eval instructions
agent_instructions = '''
You are in Named Entity Recognition (NER). 
You will be given a a list of tokens, where each token is separated by a |. 
You should read the list as a sentence.
For every token in the sentence, you will be asked to label it with one of the following labels:

    0: Outside of a named entity,
    1: Beginning of a corporation name,
    2: Inside of a corporation name,
    3: Beginning of a creative work name,
    4: Inside of a creative work name,
    5: Beginning of a group name,
    6: Inside of a group name,
    7: Beginning of a location name,
    8: Inside of a location name,
    9: Beginning of a person name,
    10: Inside of a person name,
    11: Beginning of a product name,
    12: Inside of a product name,

Output only a list of labels.
'''
eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': agent_instructions})]
eval_prompt = 'Label the following sentence.'

# -

print(len(wnut['test']))

# Let's do a subset to start
n_train = len(wnut['test'])
idxs = np.random.choice(n_train, 100, replace=False)
wnut_filter = wnut['test'].select(idxs)
wnut_df = wnut_filter.to_pandas()
wnut_df.head()

# Put together the survey based on this subset
questions = Survey([QuestionFreeText(question_name = 'q' + str(row['id']), question_text = eval_prompt + " \n" + '|'.join(row['tokens'])) for idx, row in wnut_df.iterrows()])
survey_res = questions.by(eval_agents).by(models).run()


# Get the results
res_df = survey_res.select("answer.*").to_pandas().melt()
res_df = res_df.rename(columns = {'variable': 'question', 'value': 'answer'})
res_df['id'] = [x[len('answer') + 2:] for x in res_df['question']]
res_df = res_df[~res_df['answer'].isnull()]
res_df.head()

# +
# res_df['pred'] = res_df['answer'].apply(lambda x: x.replace("|", ",") if pd.isnull(x) else x)
merged_df = wnut_df.merge(res_df, on = 'id')

# The predictions are a bit of a mess because of the return format. Let's clean them up
preds_as_list = []
for p in merged_df['answer'].values:
    y_pred = []
    for c in str(p):
        try:
            c = int(c)
            y_pred.append(c)
        except:
            continue
    preds_as_list.append(y_pred)
merged_df['pred_lst'] = preds_as_list
merged_df.head()
# -

# For some reason some of the outputs are different lengths. I don't know why. Let's filter them out for now
merged_df['true_len'] = merged_df['ner_tags'].apply(lambda x: len(x))
merged_df['pred_len'] = merged_df['pred_lst'].apply(lambda x: len(x))
merged_filt = merged_df[merged_df['true_len'] == merged_df['pred_len']] # Something went wrong. Assume this is independently wrong for now so that dropping them isn't a problem
merged_filt.head()

# +
# Compile everything into one list
y_true = [int(y) for x in merged_filt['ner_tags'].values for y in x]
y_pred = [int(y) for x in merged_filt['pred_lst'].values for y in x]

print(len(y_true), len(y_pred)) 

# +
from sklearn.metrics import f1_score

# Evaluate!
f1_score(y_true, y_pred, average = 'micro')

# This is very high... The papers on this were getting something like 30-40% F1 score. Was this dataset included in the training of gpt?
# -


