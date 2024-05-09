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
            category = f[:-4]
            cat_q = QuestionFreeText(question_name = category, question_text = query_base % category)
            qlist.append(cat_q)

    # Create the survey
    survey = Survey(questions=qlist)
    # Run
    res = survey.by(agent).by(model).run()
    # Get the responses
    return res.select("answer.*").to_pandas()


modify_instr = "You are an expert resume writer. You have been hired to edit resumes for various industries."
agent = Agent(traits={'role':"Resume Writer", 'persona':modify_instr})
model = Model("gpt-4-1106-preview")
query_base = "Edit the following resume. You can only three changes. Your output should be the updated resume with your change incorporated.\n\n %s"

def modify_text(base_dir, agent, model):
    pool = TextPool(base_dir, 'resumes', embed = False)
    qlist = []
    for t in pool.texts:
        qlist.append(QuestionFreeText(question_name = t.text_name, question_text = query_base % t.text))
    
    survey = Survey(questions=qlist)
    res = survey.by(agent).by(model).run()
    return res.select("answer.*").to_pandas()

def write_to_dir(dir, res_df, suffix = ''):
    if os.path.exists(dir):
        pass
    else:
        print(f"creating directory {dir}")
        os.makedirs(dir)

    cols = res_df.columns
    for c in cols:
        n = c.split('.')[1]
        fname = f'{dir}/{n}{suffix}.txt'
        with open(fname, 'w') as f:
            resume_text = res_df[c].values[0]
            if not pd.isnull(resume_text):
                f.write(resume_text)
            # f.write(res_df[c].values[0])
    return 

# Get the AI resumes
# ai_resumes = get_texts(resume_dir, agent, model)
# ai_resumes.to_csv('data/temp_results/ai_resumes.csv', index = False)

# # We ran the above lines so we can just read in the results
# ai_resumes = pd.read_csv('data/temp_results/ai_resumes.csv')
# write_to_dir('data/ai_resumes_test/', ai_resumes)


def dist_to_ai(pool : TextPool, print_dists : bool = False):
    
    # Separation within pool
    ai_sep = pool.calc_separation()
    print(f"resumes separation: {ai_sep}")
    
    dists = []
    for r in nonai.texts:
        sims = pool.get_similarities(r)
        idx = pool.text_names.index(r.text_name)
        dists.append(sims[idx])

        if print_dists:
            print(r.text_name)
            print("siimilarities")
            print(sims)
            print(ai.text_names)
            print("-----")

    return dists


def run_modification(source_dir, write_dir):
    mods = modify_text(source_dir, agent, model)
    write_to_dir(write_dir, mods)
    mod_dists = dist_to_ai(TextPool(write_dir, 'resumes'), print_dists = True)
    return mod_dists


print("Creating text pool with AI resumes")
ai = TextPool('data/ai_resumes/', 'resumes')
dists = dist_to_ai(ai, print_dists = False)

# Let's plot the embeddings
# embeddings = np.vstack((nonai.embeddings, ai.embeddings))
# names = nonai.text_names + [n + "_ai" for n in ai.text_names]
# TextPool.plot_embeddings(embeddings, names, label_points = True, kaggle = True)

# I'd like to see how far from the original we are over time
all_res = pd.DataFrame({'resume': nonai.text_names, 'distance_all_ai': dists})
print(all_res.head())

# Now, start to modify
# mods = modify_text('resumes/extracted_resumes/', agent, model)
# print(mods)
# write_to_dir('data/modified_resumes_1/', mods, '')

mod_dists = dist_to_ai(TextPool('data/modified_resumes_1/', 'resumes'), print_dists = False)
all_res['distance_mod_1'] = mod_dists
print(all_res.head())

# Do that again
# mods = modify_text('data/modified_resumes_1/', agent, model)
# write_to_dir('data/modified_resumes_2/', mods, '')
mod_dists = dist_to_ai(TextPool('data/modified_resumes_2/', 'resumes'), print_dists = False)
all_res['distance_mod_2'] = mod_dists
print(all_res.head())

# One more
# mod_dists = run_modification('data/modified_resumes_2/', 'data/modified_resumes_3/')
# all_res['distance_mod_3'] = mod_dists
mod_dists = dist_to_ai(TextPool('data/modified_resumes_3/', 'resumes'), print_dists = False)
all_res['distance_mod_3'] = mod_dists

# Why not keep going
# mod_dists = run_modification('data/modified_resumes_3/', 'data/modified_resumes_4/')
# all_res['distance_mod_4'] = mod_dists
mod_dists = dist_to_ai(TextPool('data/modified_resumes_4/', 'resumes'), print_dists = False)
all_res['distance_mod_4'] = mod_dists

print(all_res.groupby('resume').mean())