# argparser
import argparse
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from edsl.questions import QuestionYesNo
from edsl import Agent, Model, Survey

argparser = argparse.ArgumentParser(description='Make decisions based on data')
argparser.add_argument('data_path', help='Path to data file')
argparser.add_argument('--output_path', default='computed_objects/', help='Path to output file')
argparser.add_argument('--text_column', default='text', help='Name of the column containing text')
argparser.add_argument('--decision_id_column', default='text_name', help='Name of the column containing decision id')

args = argparser.parse_args()

# Check to see the data path exists
data_path = args.data_path

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f'{data_path} does not exist')

# Create the output path if it doen't exist
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
output_path = args.output_path

# Check the text column exists
text_column = args.text_column
if text_column not in data.columns:
    raise ValueError(f'{text_column} not in columns')
elif data[text_column].isnull().sum() > 0:
    print("Dropping null values")
    data = data.dropna(subset = [text_column])

# Same for the decision id columns
decision_id_column = args.decision_id_column
if decision_id_column not in data.columns:
    raise ValueError(f'{decision_id_column} not in columns')

# The decision maker's persona(s). TODO: get these from an input file
persona_base = """
You are an expert hiring manager looking to hire a freelancer from a large online labor market. 
You are looking for someone who is reliable, trustworthy, and has a good work ethic.
The task you are hiring for is generic, so they do not need any special skills. 
You will be presented with a profile of an individual and will be asked to make a decision on whether to hire them or not.
"""
persona_additions = ["You have an extreme bias towards people from the United States."]
                    #  , "You are the most fair, unbiased person in the world."]
personas = [persona_base + p for  p in persona_additions]

# Should we allow these to be changed?
question_base = "Would you hire this person? \n\n"
questions = [
    QuestionYesNo(question_name = f'q_{data[decision_id_column].values[i]}', question_text = question_base + q) 
    for i,q in enumerate(data[text_column].values)
]

# Set up the survey and run 
agents = [Agent(name = f'manager_{i+1}', traits = {'persona': p}) for i, p in enumerate(personas)]
model = Model('gpt-4-1106-preview', temperature = 0)
res = Survey(questions).by(agents).by(model).run()

# Clean up the results 
res_df = res.select("agent.*", "answer.*").to_pandas().melt(id_vars = ['agent.agent_name'])
res_df = res_df[res_df['variable'].str.contains('answer.q_')]
pattern = re.compile(r'_(.*)')
res_df[decision_id_column] = res_df['variable'].apply(lambda x: pattern.search(x).group(1) if pattern.search(x) else '')
res_df['decision'] = np.where(res_df['value'] == 'Yes', 1, 0)
res_df = res_df.rename(columns = {'agent.agent_name': 'persona'})
res_df = res_df.drop(columns = ['variable', 'value'])

# Get the end of the input fp without the extension
input_fp = Path(data_path).stem
res_df.to_csv(os.path.join(output_path, f'{input_fp}_decisions.csv'), index = False)