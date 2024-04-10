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

# %run text_helpers.py

# +
# Start with the job post
with open('posts/software_engineer_generic.txt', 'r') as file:
    job_post = file.read()

print(job_post)

# +
# We'll get the ideal GPT response for this post
agent_persona = "You are an expert resume writer."
agent = Agent(traits={'persona': agent_persona})
model = Model('gpt-4-1106-preview')
query = QuestionFreeText(question_name='gen_resume', question_text = "Write the ideal resume for the following job post: \n\n" + job_post)
resp = query.by(agent).by(model).run()
resume = resp.select("gen_resume").to_list()[0]

with open('resumes/morphing/ai_resume_generic_engineer.txt', 'w') as file:
    file.write(resume)

print("Done")

# +
# Now start from a human resume
with open('resumes/extracted_resumes/cleaned/technology_resume_cleaned.txt', 'r') as file:
    human_res = file.read()

agent_persona = "You are an expert resume writer. You have been hired to help people edit their resumes before they apply to the following job \n\n" + job_post
agent = Agent(traits={'persona': agent_persona})
model = Model('gpt-4-1106-preview')
q_base =  "Edit the following resume to make it suitable for the job post: \n\n %s"

for i in range(5):
    
query = QuestionFreeText(question_name='gen_resume', question_text = q_base % human_res)
