import pandas as pd
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