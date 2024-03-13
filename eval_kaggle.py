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

# %run helpers.py
# %run resume_experiment.py

# +
from resume_experiment import ResumeExperiment
from edsl import Agent, Model
from consts import model_to_str
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

model_strs = model_to_str.values()
print(model_strs)

# +
resumes = []
resume_dir = 'resumes/kaggle_resumes'
for f in os.listdir(resume_dir):
    if f.endswith('.txt'):
        with open(f'{resume_dir}/{f}', 'r') as file:
            resumes.append([f[:-4], file.read()])

print(len(resumes))
# -

n_resumes = 50
idxs = np.random.choice(len(resumes), n_resumes, replace=False)
resumes_filt = [resumes[i] for i in idxs]
print(len(resumes_filt))

resume_df = pd.DataFrame(resumes_filt, columns=['resume', 'contents'])
resume_df.head()

print(Model.available())

# +
models = Model.available()

# Where to write
expr_dir = 'experiments/kaggle_experiment'


with open('posts/software_engineer_generic.txt', 'r') as f:
    generic_post = f.read()

# The eval instructions
eval_instructions = '''
You are hiring manager at a tech company who wants to a hire a intro level software engineer. 
You have been given a set of resumes to evaluate.
The job description is as follows:
''' + generic_post
eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate and 10 corresponds to the best possible candidate'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in [ 'gpt-4-1106-preview', 'llama-2-70b-chat-hf',  'mixtral-8x7B-instruct-v0.1']] # Gpt models

# I just want to evaluate for now
features = {
    'eval_agents': eval_agents,
    'eval_models': eval_models,
    'eval_prompt': eval_prompt,
    'eval_options': eval_options
}
# -

print(eval_models)

kag_expr = ResumeExperiment(features, expr_dir)
kag_df = kag_expr.evaluate_resumes(['zzz'], resumes_filt)
kag_df.head()

print(len(kag_df))

# +
resumes_with_score = resume_df.copy()
score_cols = []

for m in kag_df['model'].unique():
    mod_str = model_to_str[m]
    score_col = f'score_{mod_str}'
    score_cols.append(score_col)
    model_df = kag_df[kag_df['model'] == m]
    model_df[score_col] = model_df['score']
    resumes_with_score = resumes_with_score.merge(model_df[['resume', score_col]], on='resume', how='left')

# let's just get rid of na columns because llama seems to be a little finnicky
resumes_with_score = resumes_with_score.dropna()
print(len(resumes_with_score))
resumes_with_score.head()

# +
col_combos = list(itertools.combinations(score_cols, 2))

for c in col_combos:
    s1, s2 = c[0], c[1]
    plt.hist(resumes_with_score[s1].values - resumes_with_score[s2].values, bins=20, alpha=0.5, label=s1)
    plt.title(f"{c[0]} - {c[1]}")
    # plt.ylabel(c[1])
    plt.show()

# Negative values mean that the first model is harsher
# Results suggest that GPT 4 is harshest, followed by llama, followed by mixtral
# Further question on if there's a clear difference across categories
# Also, is harsher actually better?

# +
# Now update, but let's only update with GPT 4

# The update instructions
update_instructions = 'You are an experiment resume writer who has been hired to improve resumes and tailor them to a specific job posting. The job posting is as follows: ' + generic_post
update_agents = [Agent(traits={'role': 'improver', 
                                'persona': update_instructions})]
update_prompt = 'Improve the following resume. You should output the entire resumes with your changes and improvements. Do not include anything in your output other than the resume.'
update_models = [Model(m) for m in ['gpt-4-1106-preview']] # Just the GPT models

features['update_agents']  = update_agents
features['update_models'] = update_models
features['update_prompt'] = update_prompt

kag_expr.update_params(features)

# -

kag_expr.update_resumes(['resumes/kaggle_resumes/'])
print("DONE")


