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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from edsl import Question, Survey, Model, Agent, Scenario
from edsl.questions import QuestionFreeText, QuestionYesNo, QuestionMultipleChoice

profiles = pd.read_csv('resumes/profiles_with_outcome.csv')
profiles.head()

for p in profiles['profile_overview'].sample(5, replace=False):
    print(p)
    print('---')

# +
profiles_filt = profiles[profiles['num_apps'] > 0]

# searchfor = ['translate', 'translation']

import re

searchfor = ['video editing', 'video editor', 'video editting']
pattern = re.compile('|'.join(searchfor), re.IGNORECASE)    
# searchfor = ['python', 'Python']
# profiles_filt = profiles_filt[profiles_filt['profile_overview'].str.contains('|'.join(searchfor))]
profiles_filt = profiles_filt[profiles_filt['profile_overview'].apply(lambda x: bool(pattern.search(x)))]
profiles_filt.head()

# +
# Make a smaller data set where half are from the US, half are not
dataset_size = 50
profiles_us = profiles_filt[profiles_filt['reg_country_name'] == 'United States'].sample(dataset_size // 2)
profiles_non_us = profiles_filt[profiles_filt['reg_country_name'] != 'United States'].sample(dataset_size // 2)
profiles_train = pd.concat([profiles_us, profiles_non_us])
profiles_train = profiles_train[['profile_overview', 'reg_country_name']]
profiles_train['hired'] = np.where(profiles_train['reg_country_name'] == 'United States', 1, 0)
# Hired column is random 1's and 0's
# profiles_train['hired'] = np.random.randint(0, 2, dataset_size)

test_size = 50
test_us = profiles_filt[(profiles_filt['reg_country_name'] == 'United States') & (~profiles_filt.index.isin(profiles_train.index))].sample(test_size // 2)
test_nus = profiles_filt[(profiles_filt['reg_country_name'] != 'United States') & (~profiles_filt.index.isin(profiles_train.index))].sample(test_size // 2)
profiles_test = pd.concat([test_us, test_nus])
profiles_test = profiles_test[['freelancer_key', 'profile_overview', 'reg_country_name']]
profiles_test['hired'] = np.where(profiles_test['reg_country_name'] == 'United States', 1, 0)

# Shuffle the dataset
profiles_train = profiles_train.sample(frac=1).reset_index(drop=True)
profiles_test = profiles_test.sample(frac=1).reset_index(drop=True)
profiles_train.head()

# +
agent_persona = """
The following is a set of profiles that a hiring manager has made decision on.  You will be given input of the form <profile>, Hired: <decision>. Profiles are separated by -----. 
The hiring manager has an extreme implicit bias towards candidates from the United States.
Many profile will not include the location of the candidate, but the hiring manager could use other information to infer the location of the candidate (e.g. spelling, sophistication of language, where the candidate went to college, etc.).
You will be given a new profiles and asked to predict if the hiring manager would hire the candidate.
"""
# question = "Based on the profiles and decisions, what are the key factors that the hiring manager is using to make decisions?"
question = "Based on the profiles you saw, do you think the following candidate was hired? The profile may not explicitly say where the candidate is from. You should try to infer that information based on the provided profiles. \n\n %s"

# Put all the decision together
compiled_decisions = ""
for i, row in profiles_train.iterrows():
    compiled_decisions += f"{row['profile_overview']}, Hired: {row['hired']}\n\n ----- \n\n"

# Create the scenarios to test on 
# scenarios = [Scenario({"profiles": item}) for item in [row['profile_overview'] for _, row in profiles_test.iterrows()]]
# q = QuestionFreeText(question_name="predict", question_text = compiled_decisions + question)
# q = QuestionYesNo(question_name="predict", question_text = question)
q_list = Survey([QuestionYesNo(question_name=f"predict_{row['freelancer_key']}", question_text = question % row['profile_overview']) for _, row in profiles_test.iterrows()])
agent = Agent(traits={'persona':agent_persona + compiled_decisions})
model = Model('gpt-4-1106-preview')

# res = q.by(scenarios).by(agent).by(model).run()
res = q_list.by(agent).by(model).run()
print(res)

# +
# print(res.select("understand").print(format='rich'))

# +
# Import confusion matrix package
from sklearn.metrics import confusion_matrix

res_df = res.select("answer.*").to_pandas().melt()
res_df = res_df[~res_df['variable'].str.contains('_comment')]
res_df['freelancer_key'] = res_df['variable'].str.split('_').str[-1].astype(int)

eval_df = profiles_test.merge(res_df, on='freelancer_key')
eval_df['pred_hired'] = (eval_df['value'] == 'Yes').astype(int)
eval_df['correct'] = (eval_df['pred_hired'] == eval_df['hired']).astype(int)
print(confusion_matrix(eval_df['hired'], eval_df['pred_hired']))
# -

for i, p in eval_df.iterrows():
    print(p['profile_overview'])
    print(p['hired'], p['pred_hired'])
    print(p['freelancer_key'])
    print('-----')

fk = '60924953'
res_with_comment = res.select("answer.*").to_pandas().melt()
res_with_comment[res_with_comment['variable'].str.contains(fk + '_comment')]['value'].values

# Why is the LLM so bad at figuring this out?
#
# Part of this is a problem with the target variable. It's really not easy to predict whether someone is from the US or not. 
# When the LLM cannot, it defaults to experiences and proficiency based on the list of skills.
# The tasks we've tried are translation and python. 
# These tasks might be part of the problem. Translation profiles usually include some mentions of the languages that they speak so the model is quick to think they are not American. 
# Python is might be very easy to override the bias because you can write down a long list of programming skills that (a) crowd out bias and (b) are the same regardless of language. 
# What task makes sense here? 
#
# I tried video editing - that seems like a good option.
# Confusion matrix is better - we get over 50% accuracy now. 
# The comparison still isn't super fair. 
# Some of these profile are nearly impossible for me to figure out if the person is from a particular country or not. 
# Maybe we should try and figure out what a different model could do (or if we had human labels that would be great too). 
# Another option that we could try is just to see when the LLM is actually confident in it's ability to determine a person's nationality. 

# +
infer_persona = """
The following is a set of candidate profiles. You will be given input of the form <profile>, Is American: <1 if American>. Profiles are separated by -----. 
Your goal is to understand whether or not a person is from the United States.
Many profile will not include the location of the candidate, but you should use other information to infer the location of the candidate (e.g. spelling and grammar, sophistication of language, where the candidate went to college, name of candidate etc.).
You will be given a new profiles and asked to predict if the candidate is from the United States.
"""
# question = "Based on the profiles and decisions, what are the key factors that the hiring manager is using to make decisions?"
infer_question = "Based on the profiles you saw, do you think the following candidate is American? If you are not confident enough, your answer should be 'Unable to determine', but only use this option if you have absolutely no idea. Otherwise, make your best guess. \n\n %s"

# Create the scenarios to test on 
infer_q_list = Survey([QuestionMultipleChoice(question_name=f"predict_{row['freelancer_key']}", question_text = infer_question % row['profile_overview'], question_options=['Yes', 'No', 'Unable to determine']) for _, row in profiles_test.iterrows()])
infer_agent = Agent(traits={'persona':infer_persona + compiled_decisions})

# res = q.by(scenarios).by(agent).by(model).run()
infer_res = infer_q_list.by(infer_agent).by(model).run()
print(infer_res)

# +
infer_df = infer_res.select("answer.*").to_pandas().melt()
infer_df = infer_df[~infer_df['variable'].str.contains('_comment')]
infer_df['freelancer_key'] = infer_df['variable'].str.split('_').str[-1].astype(int)
print(infer_df, infer_df['value'].value_counts())

eval_infer = profiles_test.merge(infer_df, on='freelancer_key')
# eval_infer = eval_infer[eval_infer['value'] != 'Unable to determine']
eval_infer['pred_hired'] = (eval_infer['value'] == 'Yes').astype(int)
eval_infer['correct'] = (eval_infer['pred_hired'] == eval_infer['hired']).astype(int)
print(eval_infer)
print(confusion_matrix(eval_infer['hired'], eval_infer['pred_hired']))
# -

for i, p in eval_infer.iterrows():
    print(p['profile_overview'])
    print(p['hired'], p['value'])
    print(p['freelancer_key'])
    print(infer_res.select(f"answer.predict_{p['freelancer_key']}_comment").print(format='rich'))
    print('-----')

# It feels like there's a lot of implicit knowledge here that isn't being exploited. 
# What if we ask the LLM to improve the resume and look at the distance between the two?

# +
# improve_persona = "You are an expert in hiring practices and have been asked to improve candidate profiles."
improve_persona = "You are an expert editor. You have been asked to fix spelling and grammar errors in candidate profiles."
improve_question = "Fix the following profile \n\n %s"
improve_qs = Survey([QuestionFreeText(question_name=f"improve_{row['freelancer_key']}", question_text = improve_question % row['profile_overview']) for _, row in profiles_test.iterrows()])
improve_res = improve_qs.by(Agent(traits={'persona':improve_persona})).run()

improve_df = improve_res.select("answer.*").to_pandas().melt()
improve_df = improve_df[~improve_df['variable'].str.contains('_comment')]
improve_df['freelancer_key'] = improve_df['variable'].str.split('_').str[-1].astype(int)
print(improve_df)
# -

from InstructorEmbedding import INSTRUCTOR
embedding_model = INSTRUCTOR('hkunlp/instructor-large')
# Merge with the originals
improve_eval = profiles_test.merge(improve_df, on='freelancer_key')
# Some profiles failed
improve_eval = improve_eval.dropna()
print(improve_eval)
# Embed the columns
original_embeds = [embedding_model.encode(x) for x in improve_eval['profile_overview'].values]
improve_embeds = [embedding_model.encode(x) for x in improve_eval['value'].values]

# +
# Calculate the cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = np.diag(cosine_similarity(original_embeds, improve_embeds))
print(similarity.shape)

plt.figure()
plt.hist(similarity.flatten(), bins=20)
plt.show()

# +
# Pretty much what I would've expected - there's a few profiles that get a lot of improvement, most aren't that far apart in embedding space. 

improve_eval['pred_hire'] = [1 if x > np.median(similarity) else 0 for x in similarity.flatten()]
print(confusion_matrix(improve_eval['hired'], improve_eval['pred_hire'])) # Not great, but not terrible. We get over 50% accuracy just based on how much the profile changed 
# -

# What was the point of this? 
# I was surprised that the LLM refused to make a decision on so many cases. 
# Reading through them, it does feel like there is signal on whether the person is US or not (I definitely could tell based on grammar, which seems like a pretty easy thing for the LLM to understand)
# My hypothesis was that if the LLM thinks there is a lot to fix in order to improve the profile, then that's a better indiciation that the profile is non-American. 
# I should've told it to just fix spelling and grammar. 
# Going back to redo that now. 
#
# Result is just about the same: we're able to get a good bit of signal out of just improving spelling and grammar 
#
# What are the next steps here? Ideally there's a nicer way to learn country of origin. 
# One of the problems that I'm running into is that we have both selective labels and omitted payoff bias. 
#
# Step 1: Filter to profiles that are relatively easy to decipher: could you infer whether this candidate is from the US or not yes/no.
# Step 2: Select a small sample to use as examples for the agent. 
# Step 3: ask to predict country
# Step 4: redact
# Step 5: ask to predict country 
#
# This is much more inline with the Tiresias work. 
# We actually want something much more general for our purposes. 
#
# For a given employer, can we learn if they will hire a person or not. 
#
# Step 1: Take the filtered dataset from above (the one where the LLM thinks it can infer race). Assign 1 to all US people in that dataset.
# Step 2: Ask the LLM to learn the decisions on a subset of the data in step 1
# Step 3: On a new dataset, ask the LLM to predict hiring on a test set
# Step 4: Ask the LLM to modify the test set based on what it learned in Step 2
# Step 5: ask the LLM to predict hiring on the the data in Step 4. 
#
# There are some clear questions that we would need to answer. 
# Does the LLM predicting hiring actually mirror what the human would have done? 
# Can't really answer that here - the answer is yes if it's able to predict country of origin but this is super stylized. 
# We would need a bunch of employer's decisions for a particular job/jobs.
# The other problem to content with is that well maybe the employer just made the right decision in not hiring this person. 
# How to get at this? 

#
