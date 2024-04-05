import os
import numpy as np

NUM_FILES = 20

# Create the directory if it doesn't exist
fp = 'resumes/kaggle_small'
if not os.path.exists(fp):
    os.makedirs(fp)
else:
    # Clear the directory if it exists
    for f in os.listdir(fp):
        os.remove(f'{fp}/{f}')

# Copy a subset of the files over
files = os.listdir('resumes/kaggle_resumes')
choices = np.random.choice(files, NUM_FILES, replace=False)
for f in choices:
    with open(f'resumes/kaggle_resumes/{f}', 'r') as file:
        text = file.read()
    with open(f'{fp}/{f}', 'w') as file:
        file.write(text)