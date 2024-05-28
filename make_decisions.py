# argparser
import argparse
import pandas as pd
import numpy as np
import os
from edsl.question import QuestionYesNo
from edsl import Agent, Model

argparser = argparse.ArgumentParser(description='Make decisions based on data')
argparser.add_argument('data_path', help='Path to data file')
argparser.add_argument('output_path', default='computed_objects/' help='Path to output file')
argparser.add_argument('text_column', default='text', help='Name of the column containing text')

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

# The decision maker's persona(s)
persona_base = """
You are an expert hiring manager looking to hire a freelancer from a large online labor market. 
You will be presented with a profile of an individual and will be asked to make a decision on whether to hire them or not.
"""
persona_additions = ["You have an extreme bias towards people from the United States"]
personas = [persona_base + p for  p in persona_additions]

question_base = "Would you hire this person? \n\n"
questions = [
    QuestionYesNo(question_name = f'q_{data.index.values[i]}', question_text = question_base + q) 
    for i,q in enumerate(data[text_column].values)
]

agents = [Agent(name = f'manager_{i+1}', traits = {'persona': p}) for i, p in enumerate(personas)]
model = Model('gpt-4-1106-preview')
res = question.by(agents).by(model).run()
