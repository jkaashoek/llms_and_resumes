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
# Let's see what this did in embedding space
from text_helpers import TextPool, Resume
from consts import *


# +
# The original profiles
samp_fp = 'resumes/profiles_sample.csv'
# The folder with updates
update_fp = 'resumes/profile_chunks/'

# Read in the originals
profiles = pd.read_csv(samp_fp)
print(profiles.head())

# Load the update
updates = []
for f in os.listdir(update_fp):
    if f.endswith('.csv'):
        update = pd.read_csv(update_fp + f, index_col=0)
        updates.append(update)

llm_profiles = pd.concat(updates)
llm_profiles.head()
# -

# We'll need to clean 
llm_profiles = llm_profiles[(llm_profiles['variable'].str.contains('answer')) & ~(llm_profiles['variable'].str.contains('_comment'))]
llm_profiles['update_type'] = [x[x.find('.') + 1 : x.find('_')] for x in llm_profiles['variable']]
llm_profiles['freelancer_key'] = [int(x.split('_')[-1]) for x in llm_profiles['variable']]
res_pivot = (llm_profiles
             .pivot(index='freelancer_key', columns='update_type', values='value')
             .reset_index())
res_pivot.head()

# +
# Count the nans in each columns
print(len(res_pivot))
res_pivot.isnull().sum()

# There's a surprising number of these. I don't understand why the llm consistently give so many. I'm going to try and patch these.
# -

# How many have everything filled in?
res_all = res_pivot.dropna()
profile_with_updates = profiles.merge(res_all, on='freelancer_key', how='inner')
profile_with_updates = profile_with_updates.rename(columns={'profile_overview':'original'})
print(len(profile_with_updates))
profile_with_updates.head()

profile_with_updates['hired'].value_counts() # We lost 37 hires. Tough - just going to keep going. We can least predict country.

# +
# There will be four pools
pool_strs = ['original', 'clean', 'redact', 'improve']
pools = [TextPool(profile_with_updates, 'resumes', text_col=pool_str, text_name_col='freelancer_key') for pool_str in pool_strs] 

print("done embedding pool")
# -

print("embedding model", embedding_model)

# +
from scipy.spatial.distance import euclidean, cosine
dist_func = euclidean

seps = []
for i, p in enumerate(pools):
    sep, sd = p.calc_separation(dist_func, add_bootstrap = True)
    seps.append(sep)
    print(f"{pool_strs[i]} separation: {sep:.3f} ({sd:.4f})")

# +
# I want to find two texts that are approximately the DiD's apart
diff_in_sep = seps[0] - seps[-1]
print(diff_in_sep)

# Find the two texts that are closest to this difference
orig_pool = pools[0]
compare_pool = pools[-1]
for i in range(len(orig_pool.texts)):
    for j in range(0, len(compare_pool.texts)):
        dist = dist_func(orig_pool.texts[i].embedding, compare_pool.texts[j].embedding)
        if abs(dist - diff_in_sep) < 1:
            print(f"Found two texts that are {dist} apart")
            print(orig_pool.texts[i])
            print("\n\n-----\n\n")
            print(compare_pool.texts[j])
            break
    else:
        continue

# +
# Let's plot them 

# Just pick two for now:
# print(original_pool.embeddings.shape)
t1 = 'original'
t2 = 'improve'
p1 = pools[pool_strs.index(t1)]
p2 = pools[pool_strs.index(t2)]

embeddings = np.append(p1.embeddings, p2.embeddings, axis=0)
names = [f"{t1}_{n}" for n in p1.text_names] + [f"{t2}_{n}" for n in p2.text_names]
TextPool.plot_embeddings(embeddings, names, kaggle=True)
# -


