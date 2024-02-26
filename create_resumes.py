import pandas as pd
import numpy as np
import edsl
from edsl.questions import QuestionFreeText
from edsl import Agent, Survey, Model
from PyPDF2 import PdfReader
import os


def extract_from_pdf(write_dir : str, existing_resume_dir : str) -> None:
    directory = os.fsencode(existing_resume_dir)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"):
            reader = PdfReader(f'{existing_resume_dir}/{filename}')
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            
            with open(f'{write_dir}/{filename[:-4]}.txt', 'w') as f:
                f.write(text)
        
        else:
            continue
        
    return

def write_resumes(write_dir : str, res_df : pd.DataFrame) -> None:
    '''
    Writes resumes to a directory
    Assumes res_df is a df with each row being a survey response
    '''
    cols = res_df.columns[:-1]
    for i, row in res_df.iterrows():
        model = row['model.model']
        for c in cols:
            idx = c.find('answer') + len('answer') + 1
            fname = f'{write_dir}/{model}_{c[idx:]}.txt'
            with open(fname, 'w') as f:
                f.write(row[c])
    return

def generate_resume(model_list : list[edsl.language_models], prompts: list[str], prompt_strs : list[str]) -> None:
    '''
    Generate a resume using a generative model
    '''
    print("Generating resumes")
    # Run everything and let edsl handle the async
    creation_agent = Agent(traits={'role': 'drafter', 
                           'persona': 'You are a resume writer and you have been hired to create a resume for a software engineer.'})
    creation_survey = Survey(questions = [QuestionFreeText(question_name = prompt_strs[i], question_text=prompt) for i, prompt in enumerate(prompts)])
    results = creation_survey.by(creation_agent).by(model_list).run()
    res_df = results.select("model.model", "answer.*").to_pandas()
    return res_df

def run_creation(write_dir : str = 'extracted_resumes', 
                 existing_resume_dir : str = 'real_resumes',
                 model_list : list[edsl.language_models] = [],
                 model_to_str : dict = {},
                 prompts : list[str] = None,
                 prompt_strs : list[str] = None) -> None:
    '''
    Extracts text from PDFs and writes to a directory or uses generative ai to create resumes
    If using ai, files will be names '<model_str>_<i>.pdf' where i is the index of the prompt
    '''
    if os.path.exists(write_dir):
        # raise UserWarning('Write directory already exists, will overwrite files. Proceeding...')
        pass
    else:
        os.makedirs(write_dir)
    
    if existing_resume_dir is not None:
        extract_from_pdf(write_dir, existing_resume_dir)
    else:
        if len(model_list) == 0:
            raise Exception('No existing resumes and no models provided. Please specify one or the other')
            return
        
        elif len(prompts) == 0:
            raise Exception('No prompts provided. Please specify a prompt for the generative model')
            return
        
        elif len(prompts) != len(prompt_strs):
            raise Exception('Number of prompts and prompt strings do not match. Please provide a prompt string for each prompt')
            return
        
        else:
            res_df = generate_resume(model_list, prompts, prompt_strs)
            res_df['model.model']  = res_df['model.model'].apply(lambda x: model_to_str[x])
            write_resumes(write_dir, res_df)
            return res_df

    return
        

# The existing resumes case
# print(run_creation(existing_resume_dir = 'real_resumes'))

# The generative model case
# models = Model.available()
# models = models[:1]
# model_objs = [Model(model) for model in models]
# model_strs = ['gpt35']#, 'gpt4', 'gemini', 'llama']
# # prompts = ['create a resume for a software engineer. Your resumes should be 1 page long and include your education, work experience, and skills',]
# prompts = ['why is the sky blue?']
# res_df = run_creation(existing_resume_dir = None,
#              model_list = model_objs, 
#              model_to_str = dict(zip(models, model_strs)), 
#              prompts = prompts, 
#              prompt_strs = ['software_engineer_resume'])
# print("Outside of function call", res_df)
# print(res_df)

extract_from_pdf('extracted_resumes', 'resumes')