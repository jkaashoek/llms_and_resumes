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

model_strs = model_to_str.values()
# print(model_strs)

# +
models = Model.available()

# Where to write
expr_dir = 'experiments/test_experiments'

# The update instructions
update_instructions = 'You are an experiment resumer writer who has been hired to improve resumes.'
update_agents = [Agent(traits={'role': 'improver', 
                                'persona': update_instructions})]
update_prompt = 'Improve the following resume.'
update_models = [Model(m) for m in models[:2]] # Just the GPT models

# The eval instructions
eval_instructions = 'You are hiring manager at a tech company who wants to a hire a intro level software engineer. You have been given a set of resumes to evaluate.'
eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate and 10 corresponds to the best possible candidate.'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models[:2]]

features = {
    'update_agents': update_agents,
    'update_models': update_models,
    'update_prompt': update_prompt,
    'eval_agents': eval_agents,
    'eval_models': eval_models,
    'eval_prompt': eval_prompt,
    'eval_options': eval_options
}
# -

# Create and run 
experiment = ResumeExperiment(features, expr_dir)
res_df = experiment.run_experiment(['resumes/extracted_resumes'])
res_df.head()


# +
def clean_evals(df):
    df['eval_model'] = df['model'].apply(lambda x: model_to_str[x])
    df['create_model'] = [x.split('_')[-1] if x.split('_')[-1] in model_strs else "from_pdf" for x in df['model'].values]

    fix_cols =  np.array([
        [x[x.find('_') + 1:], x.split('_')[0]]
        if x.split('_')[0] in model_strs
        else [x, 'no_update']
        for x in df['resume'].values
    ])

    df['resume'] = fix_cols[:,0]
    df['update_model'] = fix_cols[:,1]
    return df

def add_model_results(df, ax):
    nupdates = df['update_model'].nunique()
    width = 0.2
    for i, m in enumerate(df['update_model'].unique()):
        updated = df[df['update_model'] == m].sort_values(by='resume')
        xs = np.arange(len(updated))
        ax.bar(xs - width * (nupdates - i - 2), updated['score'], width, label=m)
        rnames = updated['resume'].values

    ax.set_xticks(xs)
    ax.set_xticklabels(rnames, rotation=45)
    ax.set_ylim([0,10])
    ax.legend()
    return ax
    
def plot_evals(df):
    all_res = df['resume'].unique()
    mods = df['eval_model'].unique()
    nmodels = len(mods)

    if nmodels < 4:
        ncols = nmodels
    else:
        ncols = 3
    
    nrows = int(np.ceil(nmodels/ncols))
    f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5))
    axs = axs.flatten()
    for i, m in enumerate(df['eval_model'].unique()):
        ax = axs[i]
        ax = add_model_results(df[df['eval_model'] == m], ax)
        ax.set_title(f"Evaluator {m}")

    plt.show()
        
    return 



# -
cleaned_evals = clean_evals(res_df.copy())
# print(cleaned_evals)
plot_evals(cleaned_evals)


# +
models = Model.available()

# Where to write
expr_dir = 'experiments/test_experiments_2'

# The update instructions
update_instructions = 'You are an experiment resumer writer who has been hired to improve resumes.'
update_agents = [Agent(traits={'role': 'improver', 
                                'persona': update_instructions})]
update_prompt = 'Improve the following resume. You should output the entire resumes with your changes and improvements. Do not include anything in your output other than the resume.'
update_models = [Model(m) for m in models[:2]] # Just the GPT models

# The eval instructions
eval_instructions = 'You are hiring manager at a tech company who wants to a hire a intro level software engineer. You have been given a set of resumes to evaluate.'
eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate and 10 corresponds to the best possible candidate.'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models[:2]]

features = {
    'update_agents': update_agents,
    'update_models': update_models,
    'update_prompt': update_prompt,
    'eval_agents': eval_agents,
    'eval_models': eval_models,
    'eval_prompt': eval_prompt,
    'eval_options': eval_options
}
# -

# Create and run 
experiment = ResumeExperiment(features, expr_dir)
res_df = experiment.run_experiment(['resumes/extracted_resumes'])
res_df.head()

cleaned_evals_2 = clean_evals(res_df.copy())
# print(cleaned_evals)
plot_evals(cleaned_evals_2)
