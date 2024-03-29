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
print(model_strs)
print(model_to_str.keys())

# +
# models = Model.available()
models = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'llama-2-70b-chat-hf', 'mixtral-8x7B-instruct-v0.1']
# Where to write
expr_dir = 'experiments/test_experiments'

# The update instructions
update_instructions = 'You are an experiment resume writer who has been hired to improve resumes.'
update_agents = [Agent(traits={'role': 'improver', 
                                'persona': update_instructions})]
update_prompt = 'Improve the following resume.'
update_models = [Model(m) for m in models] # Just the GPT models

# The eval instructions
eval_instructions = 'You are hiring manager at a tech company who wants to a hire a intro level software engineer. You have been given a set of resumes to evaluate.'
eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate and 10 corresponds to the best possible candidate'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models]

# print(models)

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


def view_comments(res_df):
    for i, row in res_df.iterrows():
        print(f"Resume {row['resume']}, model: {row['model']}, score: {row['score']}, comment: {row['comment']}")
    return


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
    width = 0.1
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
    f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
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
update_instructions = 'You are an experiment resume writer who has been hired to improve resumes.'
update_agents = [Agent(traits={'role': 'improver', 
                                'persona': update_instructions})]
update_prompt = 'Improve the following resume. You should output the entire resumes with your changes and improvements. Do not include anything in your output other than the resume.'
update_models = [Model(m) for m in models] # Just the GPT models

# The eval instructions
eval_instructions = 'You are hiring manager at a tech company who wants to a hire a intro level software engineer. You have been given a set of resumes to evaluate.'
eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate and 10 corresponds to the best possible candidate.'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models]

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
print("These are bad updates")
plot_evals(cleaned_evals)

# +
# Now let's evaluate based on a slightly different prompt

# Where to write
expr_dir = 'experiments/test_experiments_generic_post'

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
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate for this job and 10 corresponds to the best possible candidate for this job'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models[:2]]

features_generic = {
    # 'update_agents': update_agents,
    # 'update_models': update_models,
    # 'update_prompt': update_prompt,
    'eval_agents': eval_agents,
    'eval_models': eval_models,
    'eval_prompt': eval_prompt,
    'eval_options': eval_options
}
# -

# Create and run 
experiment_generic = ResumeExperiment(features_generic, expr_dir)
res_df_generic = experiment_generic.evaluate_resumes(['resumes/extracted_resumes', 
                                              'experiments/test_experiments_2/resumes/extracted_resumes_updated'])
res_df_generic.head()

plot_evals(cleaned_evals_2)
plot_evals(clean_evals(res_df_generic.copy()))

# +
# Now let's evaluate based on a slightly different prompt

# Where to write
expr_dir = 'experiments/test_experiments_fintech_post'

with open('posts/software_engineer_finance.txt', 'r') as f:
    fintech_post = f.read()

# The eval instructions
eval_instructions = '''
You are hiring manager at a tech company who wants to a hire a intro level software engineer. 
You have been given a set of resumes to evaluate.
The job description is as follows:
''' + fintech_post

eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate for this job and 10 corresponds to the best possible candidate for this job'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models[:2]]

features_fintech = {
    # 'update_agents': update_agents,
    # 'update_models': update_models,
    # 'update_prompt': update_prompt,
    'eval_agents': eval_agents,
    'eval_models': eval_models,
    'eval_prompt': eval_prompt,
    'eval_options': eval_options
}
# -

# Create and run 
experiment_fintech = ResumeExperiment(features_fintech, expr_dir)
res_df_fintech = experiment_generic.evaluate_resumes(['resumes/extracted_resumes', 
                                              'experiments/test_experiments_2/resumes/extracted_resumes_updated'])
res_df_fintech.head()

plot_evals(cleaned_evals_2)
print("generic")
plot_evals(clean_evals(res_df_generic.copy()))
print("fintech")
plot_evals(clean_evals(res_df_fintech.copy()))

view_comments(res_df_fintech)

# +
# Where to write
expr_dir = 'experiments/test_experiments_tailored_generic'

# The update instructions
update_instructions = 'You are an experiment resume writer who has been hired to improve resumes and tailor them to a specific job posting. The job posting is as follows: ' + generic_post
update_agents = [Agent(traits={'role': 'improver', 
                                'persona': update_instructions})]
update_prompt = 'Improve the following resume. You should output the entire resumes with your changes and improvements. Do not include anything in your output other than the resume.'
update_models = [Model(m) for m in models[:2]] # Just the GPT models

# The eval instructions
eval_instructions = '''
You are hiring manager at a tech company who wants to a hire a intro level software engineer. 
You have been given a set of resumes to evaluate.
The job description is as follows:
''' + generic_post

eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate and 10 corresponds to the best possible candidate.'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models[:2]]

features_tailored_generic = {
    'update_agents': update_agents,
    'update_models': update_models,
    'update_prompt': update_prompt,
    'eval_agents': eval_agents,
    'eval_models': eval_models,
    'eval_prompt': eval_prompt,
    'eval_options': eval_options
}

# Create and run 
experiment_tailored_generic= ResumeExperiment(features_tailored_generic, expr_dir)
res_df_tailored_generic = experiment_tailored_generic.run_experiment(['resumes/extracted_resumes'])
res_df_tailored_generic.head()

# +
# Where to write
expr_dir = 'experiments/test_experiments_tailored_fintech'

# The update instructions
update_instructions = 'You are an experiment resume writer who has been hired to improve resumes and tailor them to a specific job posting. The job posting is as follows: ' + fintech_post
update_agents = [Agent(traits={'role': 'improver', 
                                'persona': update_instructions})]
update_prompt = 'Improve the following resume. You should output the entire resumes with your changes and improvements. Do not include anything in your output other than the resume.'
update_models = [Model(m) for m in models[:2]] # Just the GPT models

# The eval instructions
eval_instructions = '''
You are hiring manager at a tech company who wants to a hire a intro level software engineer. 
You have been given a set of resumes to evaluate.
The job description is as follows:
''' + fintech_post

eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate and 10 corresponds to the best possible candidate.'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models[:2]]

features_tailored_fintech = {
    'update_agents': update_agents,
    'update_models': update_models,
    'update_prompt': update_prompt,
    'eval_agents': eval_agents,
    'eval_models': eval_models,
    'eval_prompt': eval_prompt,
    'eval_options': eval_options
}

# Create and run 
experiment_tailored_fintech= ResumeExperiment(features_tailored_fintech, expr_dir)
res_df_tailored_fintech = experiment_tailored_fintech.run_experiment(['resumes/extracted_resumes'])
res_df_tailored_fintech.head()
# -

print("No updates, generic")
plot_evals(clean_evals(res_df_generic.copy()))
print("Tailored to generic")
plot_evals(clean_evals(res_df_tailored_generic.copy()))
print("Tailored to fintech ")
plot_evals(clean_evals(res_df_tailored_fintech.copy()))


