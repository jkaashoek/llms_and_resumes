import pandas as pd
from PyPDF2 import PdfReader
import os


def extract_from_pdf(write_dir : str, existing_resume_dir : str) -> None:
    '''
    Extracts text from PDFs and writes to a directory
    '''
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

def write_results(write_dir : str, res_df : pd.DataFrame, suffix = False) -> None:
    '''
    Writes a dataframe of results (usually one row per updated resume) to a directory
    Assumes res_df is a df with each row being a survey response
    '''
    if os.path.exists(write_dir):
        pass
    else:
        print(f"creating directory {write_dir}")
        os.makedirs(write_dir)

    cols = res_df.columns[:-1]
    for i, row in res_df.iterrows():
        model = row['model.model']
        for c in cols:
            idx = c.find('answer') + len('answer') + 1
            # If we're updating, we'll prepend the model name, otherwise (we're generating) we'll append it
            if suffix:
                fname = f'{write_dir}/{c[idx:]}_{model}.txt'
            else:
                fname = f'{write_dir}/{model}_{c[idx:]}.txt'

            # Write
            with open(fname, 'w') as f:
                f.write(row[c])
    return

def extract_resumes_from_dir(resume_dir : str) -> list[list[str, str]]:
    '''
    Extracts the text from a directory of resumes.
    These have to be txt files containing the text extracted from a resume, not PDFs
    Results is of the form [[filename, text], ...]
    '''
    resumes = []
    for f in os.listdir(resume_dir):
        if f.endswith('.txt'):
            with open(f'{resume_dir}/{f}', 'r') as file:
                resumes.append([f[:-4], file.read()])
    return resumes

def extract_resumes_from_dir_list(dir_list : list[str]) -> list[list[str, str]]:
    '''
    Extracts the text from a list of directories of resumes.
    These have to be txt files containing the text extracted from a resume, not PDFs
    Results is of the form [[filename, text], ...]
    '''
    resumes = []
    for d in dir_list:
        resumes.extend(extract_resumes_from_dir(d))
    return resumes

# test_ai_generation()
# test_update()
# res_df = test_eval()

# extract_from_pdf('extracted_resumes', 'resumes')
    
    # def run_creation(write_dir : str = 'extracted_resumes', 
#                  existing_resume_dir : str = 'real_resumes',
#                  model_list : list[edsl.language_models] = [],
#                  model_to_str : dict = {},
#                  prompts : list[str] = None,
#                  prompt_strs : list[str] = None) -> None:
#     '''
#     Extracts text from PDFs and writes to a directory or uses generative ai to create resumes
#     If using ai, files will be names '<model_str>_<i>.pdf' where i is the index of the prompt
#     '''
#     if os.path.exists(write_dir):
#         # raise UserWarning('Write directory already exists, will overwrite files. Proceeding...')
#         pass
#     else:
#         os.makedirs(write_dir)
    
#     if existing_resume_dir is not None:
#         extract_from_pdf(write_dir, existing_resume_dir)
#     else:
#         if len(model_list) == 0:
#             raise Exception('No existing resumes and no models provided. Please specify one or the other')
#             return
        
#         elif len(prompts) == 0:
#             raise Exception('No prompts provided. Please specify a prompt for the generative model')
#             return
        
#         elif len(prompts) != len(prompt_strs):
#             raise Exception('Number of prompts and prompt strings do not match. Please provide a prompt string for each prompt')
#             return
        
#         else:
#             res_df = generate_resume(model_list, prompts, prompt_strs)
#             res_df['model.model']  = res_df['model.model'].apply(lambda x: model_to_str[x])
#             write_resumes(write_dir, res_df)
#             return res_df

#     return