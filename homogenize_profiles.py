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

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from edsl import Model, Survey, Agent
from edsl.questions import QuestionFreeText
import langdetect
import re

np.random.seed(100)
# -

# Take a look at the data

profiles = pd.read_csv("resumes/profiles_with_outcome.csv", index_col=0)
print(profiles['profile_overview'].values[6])
profiles.head()

# Let's get a sense of this data
profiles.describe()

# +
# How many hires do we have
print(profiles["hired"].value_counts())

# 3000 hires, 119k non-hires, that's a 2.5% hire rate
print(profiles["hired"].value_counts(normalize=True))
# -

profiles['reg_country_name'].value_counts()

# These are new applicants who haven't applied to anything. Let's filter to applicants who have applied to at least n places.
# I'm not sure what to make of people who haven't applied to anything. It's probably not a random sample of people who haven't applied but its unclear how exactly they'd be biased.
# Are they busier and haven't had time? Or maybe they don't see anything they're a good fit for --> likelihood they get hired is low regardless.

# +
APP_THRESHOLD = 1
profiles_filt = profiles[profiles['num_apps'] >= APP_THRESHOLD]


# Subtitute out non-character
regex = re.compile('[^a-zA-Z]')

# I also don't want to consider profiles that are written in another language. This takes a little bit of time to run: ~90 seconds
langs = []
for i, row in profiles_filt.iterrows():
    try: 
        lang = langdetect.detect(row['profile_overview'])
    except:
        lang = None
    langs.append(lang)

        
profiles_filt['lang'] = langs
profiles_filt = profiles_filt[profiles_filt['lang'] == 'en']

print(profiles_filt['hired'].value_counts(normalize=True)) # 5.5% hire rate now

# Let's see how many profiles we have left
profiles_filt.shape

# +
## UNCOMMENT THIS TO RESAMPLE

# Now sample what remains
# NUM_PROFILES = 1000
# profiles_sample = profiles_filt.sample(NUM_PROFILES, replace=False)

# Write this file so that we can keep the same sample for the next notebook
# profiles_sample.to_csv("resumes/profiles_sample.csv", index=False)
# -

profiles_sample = pd.read_csv("resumes/profiles_sample.csv")
profiles_sample['hired'].value_counts() # Not shocking, we have 71 hires

# +
# Let's run these through our LLM'
cleaning_persona = "You are an expert in creating profiles for people looking for a job in an online labor market. You will be given a set of profile and asked to format them nicely. You should not change anything about the text other than any spelling or grammar mistakes and formatting issues."
redacting_persona = "Your goal is to assist in race-blind hiring. You will be given a profile and asked to edit any information in the profile that a hiring manager could use to infer the race of the candidate. You can only edit the text to remove any such information or to format the text more nicely. Do not make any other changes to the text. "
improving_persona = "You are an expert in creating profiles for people looking for a job in an online labor market. You will be given a set of profiles. Your must improve the profile without making up any information."
question_base = "Edit the following profile. Do not include anything other than the editted text in your response.\n\n %s"

# Now define the agents 
cleaning_agent = Agent(name = 'cleaner', traits={'persona': cleaning_persona})
redacting_agent = Agent(name = 'redactor', traits={'persona': redacting_persona})
improving_agent = Agent(name = 'improver', traits={'persona': improving_persona})

# Put together all our questions
question_list = []
for i, row in profiles_sample.iterrows():
    question = QuestionFreeText(question_name = f"profile_{row['freelancer_key']:.0f}", question_text = question_base % row['profile_overview'])
    question_list.append(question)

# Now create the survey
survey = Survey(questions=question_list)
# -

# Run the survey. This should be fun with 1000 resumes.
# Take about xx minutes to run with 1k profiles.
model = Model('gpt-4-1106-preview')
res = survey.by(model).by([cleaning_agent, redacting_agent, improving_agent]).run()

res.select('agent.*', 'answer.*').to_pandas()

# Format the results
res_df = res.select('agent.*', 'answer.*').to_pandas().melt(id_vars=['agent.agent_name'])
res_df = res_df[~(res_df['variable'].str.contains('_comment')) & (res_df['variable'].str.contains('profile'))]
res_pivot = (res_df
             .pivot(index='variable', columns='agent.agent_name', values='value')
             .reset_index())
res_pivot['freelancer_key'] = [int(x.split('_')[-1]) for x in res_pivot['variable'].values]
res_pivot = res_pivot.drop(columns=['variable'])
res_pivot.head()

profile_samp_merge = profiles_sample.merge(res_pivot, on='freelancer_key')
profile_samp_merge.head()

profile_samp_merge.to_csv("resumes/profiles_with_updates.csv", index=False)


