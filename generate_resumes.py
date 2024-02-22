# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from edsl import Model, Agent
from edsl.questions import QuestionFreeText, QuestionLinearScale
import matplotlib.pyplot as plt

Model.available()

# +
model_names = {
    'gpt-3.5-turbo': 'gpt35',
    'gpt-4-1106-preview': 'gpt4',
    'llama-2-13b-chat-hf': 'llama',
    'gemini_pro': 'gemini',
}

models = [Model(x, cache=False) for x in model_names.keys()]
# -

# Just mapping out what I want to do quickly:
#
# - Generate a baseline resume using each model
#
# Two things I want to test
# - General "improvements" to a resume
# - Tuning to a specific job post
# Start with the first and then move on. That means we don't need a job post to start

# +
# Setting up our questions

# First, our drafting question
draft_question = """
Draft a detailed executive summary for a resume for an average software engineer. 
Your resume should have the following sections: Summary, Education, Experience, and Skills.
Limit your answer to the resume only.
Use real school names, company names, and skills when possible.
"""
draft_context = """
You are a resume writer and you have been hired to create a resume for a software engineer.
"""

# Now our improvements
improvement_question = """
Improve the following resume by making it more detailed and professional. You may not add or remove any sections, but you may add or remove content within the sections.
{}
"""
improvement_context = """
You are resume writing expert and have been hired to improve resumes.
"""

# Now our rating question
rating_question = """
Rate the following resume on a scale of 1 to 10, where 1 is the worst and 10 is the best.
{}
"""
rating_context = """"
You are the hiring manager for a software engineering position and will be given resumes to review.
You are looking to fill a position for a introductory level software engineer.
"""

# -

Agent(traits={'role':'', 'persona':''})

# +
# Our agents
draft_agent = Agent(traits={
    'role': 'drafter',
    'persona': draft_context})

improvement_agent = Agent(traits={
    'role': 'improver',
    'persona': improvement_context})

rating_agent = Agent(traits={
    'role': 'rater',
    'persona': rating_context})

# +
# Generate our baseline resumes
q_baseline = QuestionFreeText(
    question_name = "baseline_resume",
    question_text = draft_question
)

baseline_resumes = q_baseline.by(models).run()
# -

baseline_resumes_df = baseline_resumes.to_pandas()[['model.model', 'answer.baseline_resume']]
baseline_resumes_df.rename(columns={'answer.baseline_resume': 'resume', 'model.model':'model'}, inplace=True)
baseline_resumes_df.head()

resumes_dict = baseline_resumes_df[baseline_resumes_df['resume'].notnull()].set_index('model').to_dict()['resume']
resumes_dict

for model, resume in resumes_dict.items():
    print(f"{model}: {resume}\n") 


# +
# Now improve and score
def improve(resume, model):
    q_improve = QuestionFreeText(
        question_name = "improve",
        question_text = improvement_question.format(resume)
    )	
    r_improve = q_improve.by(improvement_agent).by(model).run()
    return r_improve[0]['answer']['improve']

def score(resume, agent, model):
    q_score = QuestionLinearScale(
        question_name = "score",
        question_text = rating_question.format(resume),
        question_options = list(range(0, 11))
    )
    r_score = q_score.by(agent).by(model).run()
    return r_score[0]['answer']['score']


# +
results = []
improvements = {}

for drafting_model, resume in resumes_dict.items():
    
    for improving_model in models:
        improved_resume = improve(resume, improving_model)
        improvements[(drafting_model, improving_model.model)] = improved_resume
        
        for scoring_model in models:
                score_result = score(improved_resume, rating_agent, scoring_model)
                            
                result = {
                    'drafting_model': drafting_model,
                    'improving_model': improving_model.model,
                    'scoring_model': scoring_model.model,
                    'score': score_result,
                    'persona': rating_agent.traits['role']
                }
                results.append(result)

# +
# We should also score the baselines
for drafting_model, resume in resumes_dict.items():
    
        for scoring_model in models:
                score_result = score(resume, rating_agent, scoring_model)
                            
                result = {
                    'drafting_model': drafting_model,
                    'improving_model': np.nan,
                    'scoring_model': scoring_model.model,
                    'score': score_result,
                    'persona': rating_agent.traits['role']
                }
                results.append(result)

results_df = pd.DataFrame(results)
results_df.head()
# -

results_df['scoring_model'] = results_df['scoring_model'].fillna('-')
results_trim = results_df[results_df['score'].notnull()]
results_trim['score'] = results_trim['score'].astype(int)
results_trim.head()

results_trim.score.values

# +
ys = np.arange(len(results_trim))
xs = results_trim['score'].values
idxs = np.argsort(xs)

labs = np.array([f"{x['drafting_model']} -> {x['improving_model']} -> {x['scoring_model']}" for _, x in results_trim.iterrows()])

f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.barh(ys, xs[idxs], color='skyblue')
ax.set_yticks(ys)
ax.set_yticklabels(labs[idxs])
plt.show()
# -
# What did these improvements actually do?
items = improvements.items()
ks, vs = zip(*items)
print(ks, vs)
improvements_df = pd.DataFrame(improvements.items(), columns=['key', 'resume'])
improvements_df['drafting_model'] = improvements_df['key'].apply(lambda x: x[0])
improvements_df['improving_model'] = improvements_df['key'].apply(lambda x: x[1])
improvements_df.head()


drafting_model = 'gpt-3.5-turbo'
print(resumes_dict[drafting_model])

for _, row in improvements_df[improvements_df['drafting_model'] == 'gpt-3.5-turbo'].iterrows():
    print(f"{row['improving_model']}: {row['resume']}\n")
    print("\n\n\n")
