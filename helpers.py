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
        
    return None

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

def write_to_dir(write_dir : str, res_df : pd.DataFrame, suffix = False) -> None:
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
                f.write(str(row[c]))
    return None

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