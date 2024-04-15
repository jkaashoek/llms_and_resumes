import pandas as pd
import numpy as np
from edsl import Agent, Survey, Model
from edsl.questions import QuestionFreeText

data_fp = 'resumes/profiles_sample.csv'
profiles_sample = pd.read_csv("resumes/profiles_sample.csv")

# Going to downsample this for testing
# profiles_sample = profiles_sample.sample(10)
# print(profiles_sample)

agent_persona = "You are an expert resume writer"

clean_q = """ 
Nicely format the following text. You should not change anything about the text other than any spelling or grammar mistakes and formatting issues. 
Do not include anything other than the editted text in your response. \n\n %s """
redact_q = """
You will be given a profile and asked to edit any information in the profile that a hiring manager could use to infer the race of the candidate. 
You can only edit the text to remove any such information or to format the text more nicely. Do not make any other changes to the text.
Do not include anything other than the editted text in your response. \n\n %s
"""
improve_q = """
Improve the following profile. You may edit the text as your wish but you cannot make up any information about the candidate. \n\n %s
"""

q_list = []
for i, row in profiles_sample.iterrows():
    q_list.append(QuestionFreeText(question_name = f"clean_{row['freelancer_key']:.0f}", question_text =  clean_q % row['profile_overview']))
    q_list.append(QuestionFreeText(question_name = f"redact_{row['freelancer_key']:.0f}", question_text =  redact_q % row['profile_overview']))
    q_list.append(QuestionFreeText(question_name = f"improve_{row['freelancer_key']:.0f}", question_text =  improve_q % row['profile_overview']))

survey = Survey(questions = q_list)
agent = Agent(name = 'resume_writer', traits = {'persona': agent_persona})
model = Model('gpt-4-1106-preview')
res = survey.by(agent).by(model).run()

res_df = res.select('agent.*', 'answer.*').to_pandas().melt(id_vars=['agent.agent_name'])
res_df.to_csv('resumes/profiles_sample_with_mods.csv')
print(res_df)