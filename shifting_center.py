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
# from edsl.questions import QuestionFreeText
# Now start from a human resume
# init_fp = 'resumes/extracted_resumes/cleaned/technology_resume_cleaned.txt'
# init_fp = 'resumes/morphing/technology_resume_iteration_6.txt'
# init_fp = 'resumes/morphing/technology_not_job_specific/technology_resume_cleaned.txt'
init_fp = 'resumes/morphing/technology_not_job_specific/technology_resume_iteration_8.txt'

with open(init_fp, 'r') as file:
    init_res = file.read()

# agent_persona = "You are an expert resume writer. You have been hired to help people edit their resumes before they apply to the following job \n\n" + job_post
agent_persona = "You are an expert resume writer. You have been hired to help people edit their resumes before they apply to a job."
agent = Agent(traits={'persona': agent_persona})
model = Model('gpt-4-1106-preview')
q_base =  "Improve the following resume to make it stronger for the job post. Return the entire resume with your changes incorporated as your response.: \n\n %s"
prev_iteration = init_res

for i in range(8, 10):
    print(f"Running iteration {i}")
    query = QuestionFreeText(question_name='gen_resume', question_text = q_base % prev_iteration)
    resp = query.by(agent).by(model).run()
    new_resume = resp.select("gen_resume").to_list()[0]
    
    # Write the result to a file
    with open (f'resumes/morphing/technology_not_job_specific/technology_resume_iteration_{i+1}.txt', 'w') as file:
        file.write(new_resume)

    prev_iteration = new_resume

print("DONE")
# -

# Now let's see what the did
pool_generic = TextPool('resumes/morphing/technology_not_job_specific/', 'resumes')
pool_job_spec = TextPool('resumes/morphing/technology_job_specific/', 'resumes')

# +
print("The generic pool") 
TextPool.plot_embeddings(pool_generic.embeddings, pool_generic.text_names, label_points = True) 


print("The job specific pool")
TextPool.plot_embeddings(pool_job_spec.embeddings, pool_job_spec.text_names, label_points = True) 

# ai_resume = Resume('resumes/morphing/ai_resume_generic_engineer.txt')

# print(pool.get_similarities(ai_resume))
# print(pool.text_names)

# -


