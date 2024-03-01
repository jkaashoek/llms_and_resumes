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

print(Model.available())
model_dict['gpt-4-1106-preview'] = 'gpt4'
# model_strs = []

agent_list = [Agent(traits={'role': 'evaluator', 
                            'persona': 'You are hiring manager at a tech company who wants to a hire a software engineer. You have been given a set of resumes to evaluate.'})]
evals = evaluate_resumes(agent_list, [Model('gpt-4-1106-preview'), Model('gpt-3.5-turbo')], ['extracted_resumes', 'extracted_resumes_updated'])
# evals_gpt35 = evaluate_resumes(agent_list, [Model('gpt-3.5-turbo')], ['extracted_resumes', 'extracted_resumes_updated'])
print(evals)
print("Done evaluating")


# +
def clean_evals(df):
    df['eval_model'] = df['model'].apply(lambda x: model_dict[x])
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
    score_no_update = df[df['update_model'] == 'no_update']['score'].values
    score_with_update = df[df['update_model'] != 'no_update']['score'].values
    # print(score_no_update, score_with_update)

    xs = np.arange(len(score_no_update))
    width = 0.2

    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax.bar(xs - width/2, score_no_update, width, label='No Update')
    ax.bar(xs + width/2, score_with_update, width, label='With Update')
    ax.set_xticks(xs)
    ax.set_xticklabels(all_res, rotation=45)
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
        ax.set_title(m)

    plt.show()
        
    return ax

cleaned_evals = clean_evals(evals)
print(cleaned_evals)
# ax = plot_evals(cleaned_evals_4)
# ax.set_title("GPT 4 Evaluator")
# plt.show()

# cleaned_evals_35 = clean_evals(evals_gpt35)
# ax = plot_evals(cleaned_evals_35)
# ax.set_title("GPT 3.5 Evaluator")
# plt.show()

# -


