# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: edsl_env
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import os

kag_resumes = pd.read_csv("../resumes/kaggle_resumes.csv")
kag_resumes.head()

# Sort for easier writing
kag_resumes = kag_resumes.sort_values(by=['Category'])
print(kag_resumes['Category'].value_counts())
kag_resumes.head()

write_dir = '../resumes/kaggle_resumes/'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

# +

uniq_idx, prev_cat = 0, None
for _, row in kag_resumes.iterrows():
    cat = row['Category']

    if cat != prev_cat:
        uniq_idx = 0

    fname = f"{write_dir}{cat.replace(' ', '').lower()}_{uniq_idx}.txt"

    with open(fname, 'w') as f:
        f.write(row['Resume'])

    prev_cat = cat
    uniq_idx += 1

print("Done writing")
# -


