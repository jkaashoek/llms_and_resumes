import pandas as pd
import numpy as np
import os
from edsl import Agent, Survey, Model
from edsl.questions import QuestionFreeText

data_fp = 'resumes/profiles_sample.csv'
profiles_sample = pd.read_csv(data_fp)

# Going to downsample this for testing
# profiles_sample = profiles_sample.sample(100)
# # print(profiles_sample)

agent_persona = "You are an expert at writing profiles for a large online labor market."

clean_q = """ 
Nicely format the following text. You should not change anything about the text other than any spelling or grammar mistakes and formatting issues. 
Do not include anything other than the editted text in your response. \n\n %s """
redact_q = """
You will be given a profile and asked to edit any information in the profile that a hiring manager could use to infer the race of the candidate. 
You can only edit the text to remove any such information or to format the text more nicely. Do not make any other changes to the text.
Do not include anything other than the editted text in your response.
Here is the text: \n\n %s
"""
improve_q = """
Improve the following profile. You may edit the text as your wish but you cannot make up any information about the candidate. \n\n %s
"""

# Agent and model
agent = Agent(name = 'resume_writer', traits = {'persona': agent_persona})
model = Model('gpt-4-1106-preview')

# We'll process in chunks becuase doing 1000 at a time was really slow and I want to keep track of things as they go
chunksize = 100
i = 0
def inital_run(data_fp, chunksize):
    with pd.read_csv(data_fp, chunksize=chunksize) as reader:
        for chunk in reader:
            print("chunk", i)
            # if (i > 0):
                # break
            q_list = []
            for _, row in chunk.iterrows():
                q_list.append(QuestionFreeText(question_name = f"clean_{row['freelancer_key']:.0f}", question_text =  clean_q % row['profile_overview']))
                q_list.append(QuestionFreeText(question_name = f"redact_{row['freelancer_key']:.0f}", question_text =  redact_q % row['profile_overview']))
                q_list.append(QuestionFreeText(question_name = f"improve_{row['freelancer_key']:.0f}", question_text =  improve_q % row['profile_overview']))
            survey = Survey(questions = q_list)
            res = survey.by(agent).by(model).run()
            
            res_df = res.select('agent.*', 'answer.*').to_pandas().melt(id_vars=['agent.agent_name'])
            res_df.to_csv(f'resumes/profile_chunks/profiles_sample_with_mods_{i}.csv')
            i += 1
    return 


# inital_run(data_fp, chunksize)

def get_q_base(update_type):
    if update_type == 'clean':
        return clean_q
    elif update_type == 'redact':
        return redact_q
    elif update_type == 'improve':
        return improve_q
    else:
        return None
    
def clean_edsl_result(update_df):
    update_df = update_df[(update_df['variable'].str.contains('answer')) & ~(update_df['variable'].str.contains('_comment'))]
    update_df['update_type'] = [x[x.find('.') + 1 : x.find('_')] for x in update_df['variable']]
    update_df['freelancer_key'] = [int(x.split('_')[-1]) for x in update_df['variable']]
    # res_df = (update_df
    #             .pivot(index='freelancer_key', columns='update_type', values='value')
    #             .reset_index())
    return update_df

def get_unaswered(update_df):
     # Add the questions that weren't answered
    q_list = []
    for _, row in update_df.iterrows():
        if pd.isnull(row['value']) or row['value'] == '':
            q_list.append(QuestionFreeText(question_name = row['variable'].split('.')[-1], question_text =  get_q_base(row['update_type']) % row['profile_overview']))
    return q_list

def merge_runs(update_df, patch_df):
    # Merge the two dataframes
    update_df = update_df.merge(patch_df, on = 'variable', suffixes=('', '_patch'), how = 'left')
    # If the patch is not null, use the patch, otherwise use the original
    patched_vals = np.where(update_df['value_patch'].isnull(), update_df['value'], update_df['value_patch'])
    update_df['value'] = patched_vals
    update_df = update_df.drop(columns = ['value_patch', 'agent.agent_name_patch', 'freelancer_key_patch'], axis = 1, size = 0.5)

    return update_df

def patch_df(update_df):
    '''
    Recursively fill in the updates until all questions are answered
    '''
    # Get the inital list of questions
    q_list = get_unaswered(update_df)
    print("number unanswered", len(q_list))
    print(q_list)

    if len(q_list) < 240:
        return update_df
   
    # Run the survey             
    res = Survey(questions = q_list[:10]).by(agent).by(model).run()
    temp_df = res.select('agent.*', 'answer.*').to_pandas().melt(id_vars=['agent.agent_name'])
    res_df = clean_edsl_result(temp_df)
    print(res_df)

    # Merge the two dataframes
    update_df = merge_runs(update_df, res_df)    
    return patch_df(update_df)

# Read in the results and patch
updates = []
update_fp = 'resumes/profile_chunks/'
for f in os.listdir(update_fp):
    if f.endswith('.csv'):
        update = pd.read_csv(update_fp + f, index_col=0)
        updates.append(update)

update_df = pd.concat(updates)

# Clean this up
update_df = clean_edsl_result(update_df)
update_df = update_df.merge(profiles_sample[['freelancer_key', 'profile_overview']], on = 'freelancer_key')
print(update_df, len(update_df), len(update_df[update_df['value'].isnull()]))

patched_updates = patch_df(update_df)
patched_updates.to_csv("resumes/patched_profile_updates.csv", index=False)
# print(res_df)