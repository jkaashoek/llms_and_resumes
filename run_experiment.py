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

from helpers import *
import matplotlib.pyplot as plt

agent_list = [Agent(traits={'role': 'evaluator', 
                            'persona': 'You are hiring manager at a tech company who wants to a hire a software engineer. You have been given a set of resumes to evaluate.'})]
evals = evaluate_resumes(agent_list, model_objs, ['extracted_resumes', 'extracted_resumes_updated'])
print(evals)

evals['eval_model'] = evals['model'].apply(lambda x: model_dict[x])
fix_cols =  np.array([
    [x[x.find('_') + 1:], x.split('_')[0]]
    if x.split('_')[0] in model_strs and x.split('_')[1] not in model_strs
    else [x, 'no_update']
    for x in evals['resume'].values
])
evals['resume'] = fix_cols[:,0]
evals['update_model'] = fix_cols[:,1]
evals.head()

evals[evals['update_model'] == 'no_update']

# +
all_res = evals['resume'].unique()

score_no_update = evals[evals['update_model'] == 'no_update']['score'].values
score_with_update = evals[evals['update_model'] != 'no_update']['score'].values
print(score_no_update, score_with_update)

xs = np.arange(len(all_res))
width = 0.2

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

ax.bar(xs - width/2, score_no_update, width, label='No Update')
ax.bar(xs + width/2, score_with_update, width, label='With Update')
ax.legned()
plt.show()
# print(score_no_update)
# -


