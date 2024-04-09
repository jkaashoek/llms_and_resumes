from text_helpers import TextObj, Resume, JobDescription, TextPool
from edsl.questions import QuestionFreeText
from edsl import Model, Agent, Survey
import pandas as pd
import numpy as np
import os

# Start with our known non-AI resumes
resume_dir = 'resumes/extracted_resumes/'
print("Creating text pool with embeddings")
nonai = TextPool(resume_dir, 'resumes')
nonai_sep = nonai.calc_separation()
print(f"Non-AI resumes separation: {nonai_sep}")


# Put this in a function to limit LLM calls
# Now let's add some AI resumes that are meant to be in the same category as the non AI ones
# Go through each non-AI resume and add an AI resume that is similar
llm_agent_instr = "You are an expert resume writer. You have been hired to write resumes for various industries. Each resumes should be at the level of a currently-enrolled college Freshman or Sophomore."
agent = Agent(traits={'role':"Resume Writer", 'persona':llm_agent_instr})
query_base = "Generate a resume for a %s student. Your resume should be about a page long and include the following sections: Education, Experience, Skills, and Interests."
model = Model("gpt-4-1106-preview")

def get_texts(resume_dir, agent, model):
    qlist = []
    for f in os.listdir(resume_dir):
        if f.endswith('.txt'):
            # Get the category
            category = f.split('_')[0]
            cat_q = QuestionFreeText(question_name = category, question_text = query_base % category)
            qlist.append(cat_q)

    # Create the survey
    survey = Survey(questions=qlist)
    # Run
    res = survey.by(agent).by(model).run()
    # Get the responses
    return res.select("answer.*").to_pandas()

# Get the AI resumes
# ai_resumes = get_texts(resume_dir, agent, model)
# ai_resumes.to_csv('data/temp_results/ai_resumes.csv', index = False)

# We ran the above lines so we can just read in the results
# ai_resumes = pd.read_csv('data/temp_results/ai_resumes.csv')

# for c in ai_resumes.columns:
#     category = c.split('.')[1]
#     with open(f'data/ai_resumes/{category}_resume.txt', 'w') as f:
#         f.write(ai_resumes[c].values[0])
# print("Done")

ai = TextPool('data/ai_resumes/', 'resumes')
ai_sep = ai.calc_separation()
print(f"AI resumes separation: {ai_sep}")