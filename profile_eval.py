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
plot_args = {'s': 4, 'alpha': 0.4}
TextPool.plot_embeddings(embeddings, names, kaggle=True, **plot_args)

# +
# Now, let's see if we can predict the country
profile_with_updates['reg_country_name'].value_counts()

# Let's just try to predict if the coutnry is US or not
profile_with_updates['is_us'] = profile_with_updates['reg_country_name'] == 'United States'

# Train-test split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
import nltk


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import xgboost as xgb 

cv = CountVectorizer(stop_words='english', max_features=5000)

# We'll start pretty simply with a bag of words model
X_train, X_test, y_train, y_test = train_test_split(profile_with_updates, profile_with_updates['is_us'], test_size=0.2, random_state=42)
print(y_test.value_counts())

p_str = 'original'
for p_str in pool_strs:
    X_train_cv = cv.fit_transform(X_train[p_str]) 
    X_test_cv = cv.transform(X_test[p_str])

    # clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.1,silent=True, objective='binary:logistic', \
                        # booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                        # subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1, reg_alpha=0, reg_lambda=1)
    clf = xgb.XGBClassifier()
    clf.fit(X_train_cv, y_train)
    y_pred = clf.predict(X_test_cv)
    print(f"--- AUROC on test data for {p_str} ---")
    print(roc_auc_score(y_test, y_pred))



