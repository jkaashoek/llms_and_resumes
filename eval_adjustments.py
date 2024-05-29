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

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

base_folder = 'computed_objects/'
data_base = 'profiles_original_sample_100'
data_orig = pd.read_csv(base_folder + data_base + '_decisions' + '.csv')
data_generic = pd.read_csv(base_folder + data_base + '_generic_adjustment_decisions.csv')
data_tailored = pd.read_csv(base_folder + data_base + '_tailored_adjustment_decisions.csv')

all_data = pd.merge(data_orig, data_generic, on='freelancer_key', how='inner', suffixes=('', '_generic'))
all_data = pd.merge(all_data, data_tailored, on='freelancer_key', how='inner', suffixes=('', '_tailored'))
all_data = all_data.drop(columns=[x for x in all_data.columns if 'persona' in x])
all_data

# +
text_orig = pd.read_csv('data/sampled_profiles/' + data_base + '.csv')
text_orig = text_orig[['freelancer_key', 'profile_overview']]

generic_text = pd.read_csv(base_folder + data_base + '_generic_adjustment.csv')
tailored_text = pd.read_csv(base_folder + data_base + '_tailored_adjustment.csv')



# +
# Code to compare strings. Direct copy and paste from https://towardsdatascience.com/side-by-side-comparison-of-strings-in-python-b9491ac858

import difflib
import re

def tokenize(s):
    return re.split('\s+', s)

def untokenize(ts):
    return ' '.join(ts)
        
def equalize(s1, s2):
    l1 = tokenize(s1)
    l2 = tokenize(s2)
    res1 = []
    res2 = []
    prev = difflib.Match(0,0,0)
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        if (prev.a + prev.size != match.a):
            for i in range(prev.a + prev.size, match.a):
                res2 += ['_' * len(l1[i])]
            res1 += l1[prev.a + prev.size:match.a]
        if (prev.b + prev.size != match.b):
            for i in range(prev.b + prev.size, match.b):
                res1 += ['_' * len(l2[i])]
            res2 += l2[prev.b + prev.size:match.b]
        res1 += l1[match.a:match.a+match.size]
        res2 += l2[match.b:match.b+match.size]
        prev = match
    return untokenize(res1), untokenize(res2)

def insert_newlines(string, every=64, window=10):
    result = []
    from_string = string
    while len(from_string) > 0:
        cut_off = every
        if len(from_string) > every:
            while (from_string[cut_off-1] != ' ') and (cut_off > (every-window)):
                cut_off -= 1
        else:
            cut_off = len(from_string)
        part = from_string[:cut_off]
        result += [part]
        from_string = from_string[cut_off:]
    return result

def show_comparison(s1, s2, width=40, margin=10, sidebyside=True, compact=False):
    s1, s2 = equalize(s1,s2)

    if sidebyside:
        s1 = insert_newlines(s1, width, margin)
        s2 = insert_newlines(s2, width, margin)
        if compact:
            for i in range(0, len(s1)):
                lft = re.sub(' +', ' ', s1[i].replace('_', '')).ljust(width)
                rgt = re.sub(' +', ' ', s2[i].replace('_', '')).ljust(width) 
                print(lft + ' | ' + rgt + ' | ')        
        else:
            for i in range(0, len(s1)):
                lft = s1[i].ljust(width)
                rgt = s2[i].ljust(width)
                print(lft + ' | ' + rgt + ' | ')
    else:
        print(s1)
        print(s2)

def compare(fk):
    decisions = all_data[all_data['freelancer_key'] == fk]
    print(f"---comparing fk {fk}. Decisions: {decisions['decision'].values[0]}, {decisions['decision_generic'].values[0]}, {decisions['decision_tailored'].values[0]}")
    show_comparison(text_orig[text_orig['freelancer_key'] == fk]['profile_overview'].values[0],
                    generic_text[generic_text['freelancer_key'] == fk]['profile_overview'].values[0], sidebyside=True, compact=True)
    print('-------------------------------------------------------------')
    show_comparison(text_orig[text_orig['freelancer_key'] == fk]['profile_overview'].values[0],
                    tailored_text[tailored_text['freelancer_key'] == fk]['profile_overview'].values[0], sidebyside=True, compact=True)
    print('-------------------------------------------------------------')
