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

# %run text_helpers.py

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import os

# Our dataset 
profiles = pd.read_csv("resumes/profiles_sample_with_mods.csv")
profiles.head()

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

# How many have everything filled in?
res_all = res_pivot.dropna()
profile_with_updates = profiles.merge(res_all, on='freelancer_key', how='inner')
profile_with_updates = profile_with_updates.rename(columns={'profile_overview':'original'})
print(len(profile_with_updates))
profile_with_updates.head()

profile_with_updates.to_csv("resumes/profiles_with_updates.csv", index=False)

# There will be two pools
pool_strs = ['original', 'improve']
pools = [TextPool(profile_with_updates, 'resumes', text_col=pool_str, text_name_col='freelancer_key') for pool_str in pool_strs] 


# +
# We'll get the dimension-reduced embeddings
 # X_embedded = TSNE(n_components=2,  perplexity = min(5, embeddings.shape[0] - 1)).fit_transform(embeddings)

# How many dimensions do we want
dim = 2

original_embeddings = pools[0].embeddings
improved_embeddings = pools[1].embeddings
X_embedded = PCA(n_components=dim).fit_transform(original_embeddings)
Y_embedded = PCA(n_components=dim).fit_transform(improved_embeddings)

# Distance between points
from scipy.spatial import distance
dist_func = distance.euclidean

def objective_func(params, X_embedded, Y_embedded):
    # Extract the parameter values
    alpha = params[:dim]
    w = params[dim:2*dim]
    z = params[2*dim:3*dim]
    sigma = params[-1]

    # The objective function is y - (x - alpha * (z - x) + w + noise)
    temp = (X_embedded - alpha * (z - X_embedded) + w + np.random.normal(0, sigma, X_embedded.shape))
    objective = [dist_func(temp[i], Y_embedded[i]) for i in range(len(Y_embedded))]

    # The objective is the sum of the squares
    return np.sum(objective)


# +
# Now minimize
# We need the variance to be >0
def run_opt():
    # Our params are: alpha, the compression vector
    # alpha_0 = np.zeros(dim)
    alpha_0 = np.random.normal(size=dim)
    # w, the translation vector
    w_0 = np.random.normal(size=dim)
    # w_0 = np.zeros(dim)
    # z, the point we're being drawn towards (or away from)
    z_0 = np.random.normal(size=dim)
    # The variance of the noisy
    sigma_0 = np.random.uniform(0, 1)
    params = np.concatenate([alpha_0, w_0, z_0, [sigma_0]])
    
    return minimize(objective_func, params, args=(X_embedded, Y_embedded), bounds=[(0, None)] * dim + [(None, None)] * (2 * dim) + [(0, None)])['x']


# We'll run the optimization a bunch
sims = 100
results = [run_opt() for _ in range(sims)]
print('done')
# -

# Summary of results
results_df = pd.DataFrame(results, columns=[f'alpha_{i}' for i in range(dim)] + [f'w_{i}' for i in range(dim)] + [f'z_{i}' for i in range(dim)] + ['sigma'])
results_df.describe()

# +
import matplotlib.pyplot as plt
ctr_pt = np.mean(results, axis=0)[2*dim:3*dim]
print(ctr_pt)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], color='blue')
plt.scatter(Y_embedded[:, 0], Y_embedded[:, 1], color='red')
# plt.scatter(ctr_pt[0], ctr_pt[1], color='green')
plt.show()
# -


