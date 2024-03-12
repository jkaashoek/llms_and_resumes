from edsl.questions import QuestionFreeText, QuestionLinearScale
from edsl import Survey
from helpers import extract_resumes_from_dir, write_to_dir, extract_resumes_from_dir_list
from consts import model_to_str
import pandas as pd
import os

class ResumeExperiment:
    def __init__(self, features : dict, expr_dir : str) -> None:
        self.features = features
        self.expr_dir = expr_dir
        return

    def update_params(self, new_params) -> None:
        for k, v in new_params.items():
            self.features[k] = v
        return
    
    def update_resumes(self, update_dirs, resumes = None, write_dir_suffix = '_updated') -> pd.DataFrame:
        '''
        TODO: update and eval are very similar...  can we combine?
        '''
        if resumes is None:
            resumes = extract_resumes_from_dir_list(update_dirs)
        
        # This will fail if there are files that have the same name in different directories
        # are we okay with that? Something to keep in mind at least if we start doing this at scale.
        write_dir = f'{self.expr_dir}/updated_resumes'

        # Evaluate the resumes where the question name is the filename
        update_survey = Survey(questions = [
            QuestionFreeText(question_name = name, question_text=self.features['update_prompt'] + r_text) 
            for name, r_text in resumes
        ])
        results = update_survey.by(self.features['update_agents']).by(self.features['update_models']).run()
        res_df = results.select("model.model", "answer.*").to_pandas()
        res_df['model.model']  = res_df['model.model'].apply(lambda x: model_to_str[x])
        # Sometimes the models return a _comment column, which we don't want
        # Seems to only happen with the small llama model. 
        drop_cols = [c for c in res_df.columns if c.find('comment') != -1]
        res_df = res_df.drop(columns = drop_cols)
        write_to_dir(write_dir, res_df)
        return res_df

    def clean_eval_results(self, res_df : pd.DataFrame) -> pd.DataFrame:
        # I want a data frame with columns: MODEL, FILE (RESUME), SCORE, COMMENT
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
    
    def evaluate_resumes(self, eval_dirs : list[str], resumes = None) -> pd.DataFrame:
        if resumes is None:
            resumes = extract_resumes_from_dir_list(eval_dirs)
        # Evaluate the resumes where the question name is the filename
        eval_survey = Survey(questions = [QuestionLinearScale(question_name = name, 
                                                        question_text=self.features['eval_prompt'] + r_text, 
                                                        question_options = self.features['eval_options']) for name, r_text in resumes])
        results = eval_survey.by(self.features['eval_agents']).by(self.features['eval_models']).run()
        # TODO: write here instead of in the run_experiment function?
        return self.clean_eval_results(results.to_pandas())
    
    def run_experiment(self, update_dirs : list[str]) -> pd.DataFrame:
        self.update_resumes(update_dirs)
        # updated_folders = [f'{self.expr_dir}/{d}_updated' for d in update_dirs]
        updated_folders = [f'{self.expr_dir}/updated_resumes']
        # dirs_to_eval = update_dirs.extend(updated_folders)
        res_df = self.evaluate_resumes(update_dirs + updated_folders)
        res_df.to_csv(f'{self.expr_dir}/results.csv')
        return res_df
    

# # Example usage
# from edsl import Agent, Model
# models = Model.available()

# # Where to write
# expr_dir = 'experiments/test_experiments'

# The update instructions
update_instructions = 'You are an experiment resume writer who has been hired to improve resumes.'
update_agents = [Agent(traits={'role': 'improver', 
                                'persona': update_instructions})]
update_prompt = 'Improve the following resume.'
update_models = [Model(m) for m in models[4:5]] # Just the GPT models

# The eval instructions
eval_instructions = 'You are hiring manager at a tech company who wants to a hire a intro level software engineer. You have been given a set of resumes to evaluate.'
eval_agents = [Agent(traits={'role': 'evaluator',
                             'person': eval_instructions})]
eval_prompt = 'Evaluate the following resume on a scale from 1 to 10, where 1 corresponds to the worst possible candidate and 10 corresponds to the best possible candidate'
eval_options = list(range(0, 11))
eval_models = [Model(m) for m in models[3:4]]

# features = {
#     'update_agents': update_agents,
#     'update_models': update_models,
#     'update_prompt': update_prompt,
#     'eval_agents': eval_agents,
#     'eval_models': eval_models,
#     'eval_prompt': eval_prompt,
#     'eval_options': eval_options
# }

# Create the experiment
print(update_models)
exp = ResumeExperiment(features, expr_dir)
exp.update_resumes(['resumes/extracted_resumes'])
