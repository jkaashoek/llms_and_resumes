import pandas as pd
import numpy as np
import edsl
from edsl.questions import QuestionFreeText, QuestionLinearScale
from edsl import Agent, Survey, Model
from helpers import extract_from_pdf, write_results, txt_to_df
from consts import model_to_str
import os 


# Three parts to any experiment
# Generation 
# Update
# Evalute
# Each should have it's own agent(s), model(s), and instruction(s)

# Features:
# update_agent, update_models
# evaluate_agent, evaluate_models 

class ResumeExperiment:
    def __init__(self, features : dict, expr_dir : str) -> None:
        self.features = features
        self.expr_dir = expr_dir
        return

    def update_params(self, new_params) -> None:
        for k, v in new_params.items():
            self.features[k] = v
        return
    
    def update_resumes(self, update_dirs, write_dir_suffix = '_updated') -> pd.DataFrame:
        for d in update_dirs:
        
            if not os.path.exists(d):
                raise Exception(f'Directory {d} does not exist')
                continue

            else:
                write_dir = f'{self.expr_dir}{d}{write_dir_suffix}'

                # Reads the txt from resumes into a dataframe
                resumes = txt_to_df(d)
                # Evaluate the resumes where the question name is the filename
                update_survey = Survey(questions = [
                    QuestionFreeText(question_name = name, question_text=self.features['update_prompt'] + r_text) 
                    for name, r_text in resumes
                ])
                results = update_survey.by(self.features['agent_list']).by(self.features['model_list']).run()
                res_df = results.select("model.model", "answer.*").to_pandas()
                res_df['model.model']  = res_df['model.model'].apply(lambda x: model_to_str[x])
                write_results(write_dir, res_df)
            return res_df



def evaluate_resumes(agent_list : list[edsl.agents], 
                    model_list : list[edsl.language_models],
                    eval_dirs : list[str],
                    eval_prompt : str = "Evaluate the following resume.",
                    eval_options = list(range(0, 11))) -> pd.DataFrame:
    '''
    Evaluates a resume in a given list of directories
    '''
    resumes = []
    for d in eval_dirs:
        if not os.path.exists(d):
            raise Exception(f'Directory {d} does not exist')
        else:
            resumes.extend(extract_resumes_from_dir(d))
            
    # Evaluate the resumes where the question name is the filename
    eval_survey = Survey(questions = [QuestionLinearScale(question_name = name, 
                                                       question_text=eval_prompt + r_text, 
                                                       question_options = eval_options) for name, r_text in resumes])
    results = eval_survey.by(agent_list).by(model_list).run()
    # res_df = results.select("model.model", "answer.*").to_pandas()

    # I want a data frame with columns: MODEL, FILE (RESUME), SCORE, COMMENT
    res_df = results.to_pandas()
    # Melt down on agent and model
    res_df = res_df.melt(id_vars = ['agent.agent_name', 'model.model'], var_name = 'data', value_name = 'answer')
    # Limit to just the answers
    res_df = res_df[res_df['data'].str.contains('answer')]
    # Get the comments
    is_comment = res_df['data'].str.contains('comment')
    # Remove the answer label from the column
    res_df['data'] = [
        x[len('answer') + 1 : x.rfind('_')] 
        if x.find('comment') != -1
        else x[len('answer') + 1:]
        for x in res_df['data']
    ]
    
    # Now separate and merge
    comments = res_df[is_comment]
    scores = res_df[~is_comment]
    cleaned_res = (scores
                   .merge(comments, on = ['agent.agent_name', 'model.model', 'data'], suffixes = ('_score', '_comment'))
                   .rename(columns = {'agent.agent_name': 'agent', 'model.model':'model', 
                                      'data': 'resume', 'answer_score': 'score', 'answer_comment': 'comment'}))
    return cleaned_res



def generate_resumes(agent_list : list[edsl.agents], 
                    model_list : list[edsl.language_models],
                    model_to_str : dict,
                    prompt_dict : dict,
                    write_dir : str) -> pd.DataFrame:
    '''
    Generate a resume using a generative model
    '''
    print("Generating resumes")
    # Run everything and let edsl handle the async
    creation_survey = Survey(questions = [QuestionFreeText(question_name = k, question_text=v) for k, v in prompt_dict.items()])
    results = creation_survey.by(agent_list).by(model_list).run()
    res_df = results.select("model.model", "answer.*").to_pandas()
    res_df['model.model']  = res_df['model.model'].apply(lambda x: model_to_str[x])
    write_resumes(write_dir, res_df, suffix=True)
    return res_df




# The existing resumes case
models = Model.available()
models = models[:1]
model_strs = ['gpt35']
model_objs = [Model(model) for model in models]
model_dict = dict(zip(models, model_strs))

# The generative model case
def test_ai_generation():
    agent_list = [Agent(traits={'role': 'drafter', 
                                'persona': 'You are a resume writer and you have been hired to create a resume for a software engineer.'})]
    prompts = ['Generate a resume for a software engineer. Your resumes should be 1 page long and include your education, work experience, and skills.']
    prompt_strs = ['software_engineer_resume']
    prompt_dict = dict(zip(prompt_strs, prompts))

    res_df = generate_resumes(agent_list, model_objs, model_dict, prompt_dict, 'extracted_resumes')
    # print("Outside of function call", res_df)
    print(res_df)

def test_update():
    agent_list = [Agent(traits={'role': 'improver', 
                                'persona': 'You are an expert resume writer and have been hired to improve existing resumes.'})]
    # prompts = ['create a resume for a software engineer. Your resumes should be 1 page long and include your education, work experience, and skills',]
    prompts = ['Improve the following resume. Your resumes should be 1 page long and include your education, work experience, and skills.']
    prompt_strs = ['software_engineer_resume_1']
    prompt_dict = dict(zip(prompt_strs, prompts))

    res_df = update_resumes(agent_list, model_objs, model_dict, ['extracted_resumes'])
    print(res_df)


def test_eval():
    agent_list = [Agent(traits={'role': 'evaluator', 
                                'persona': 'You are hiring manager at a tech company who wants to a hire a software engineer. You have been given a set of resumes to evaluate.'})]
    return evaluate_resumes(agent_list, model_objs, ['extracted_resumes'])


# def update_resumes(agent_list : list[edsl.agents], 
#                     model_list : list[edsl.language_models],
#                     model_to_str : dict,
#                     update_dirs : list[str],
#                     update_prompt : str = "Improve the following resumes",
#                     write_dir_suffix = '_updated') -> pd.DataFrame:
#     '''
#     Update a resume using the agents and models provided
#     '''
#     for d in update_dirs:
        
#         if not os.path.exists(d):
#             raise Exception(f'Directory {d} does not exist')
#             continue

#         else:
#             write_dir = f'{d}{write_dir_suffix}'

#             resumes = extract_resumes_from_dir(d)
#             # Evaluate the resumes where the question name is the filename
#             update_survey = Survey(questions = [QuestionFreeText(question_name = name, 
#                                                                   question_text=update_prompt + r_text) for name, r_text in resumes])
#             results = update_survey.by(agent_list).by(model_list).run()
#             res_df = results.select("model.model", "answer.*").to_pandas()
#             res_df['model.model']  = res_df['model.model'].apply(lambda x: model_to_str[x])
#             write_resumes(write_dir, res_df)
#     return res_df