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

# +
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Distance between points
from scipy.spatial import distance
# -

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
# class AlphaExper():

#     def __init__(self, x, y) -> None:
#         self.x = x
#         self.y = y

#     def sample(self, smpl_size):
#         idxs = np.random.choice(range(len(self.x)), smpl_size)
#         return self.x[idxs], self.y[idxs]

#     def embed

# -



# +
# We'll get the dimension-reduced embeddings
 # X_embedded = TSNE(n_components=2,  perplexity = min(5, embeddings.shape[0] - 1)).fit_transform(embeddings)

original_embeddings = pools[0].embeddings
improved_embeddings = pools[1].embeddings
dist_func = distance.euclidean

def objective_func(params, X_embedded, Y_embedded, alpha_dim, dim):
    # Extract the parameter values
    alpha = params[:alpha_dim]
    w = params[alpha_dim:alpha_dim + dim]
    z = params[alpha_dim + dim:alpha_dim + 2*dim]
    sigma = params[-1]

    # The objective function is y - (x - alpha * (z - x) + w + noise)
    temp = (X_embedded + alpha * (z - X_embedded) + w + np.random.normal(0, sigma, X_embedded.shape))
    objective = [dist_func(temp[i], Y_embedded[i]) for i in range(len(Y_embedded))]

    # The objective is the sum of the squares
    return np.sum(objective)

# Now minimize
# We need the variance to be >0
def run_opt(X, Y, alpha_dim, dim):
    # Our params are: alpha, the compression vector
    # alpha_0 = np.zeros(dim)
    alpha_0 = np.random.normal(size=alpha_dim)
    # w, the translation vector
    w_0 = np.random.normal(size=dim)
    # w_0 = np.zeros(dim)
    # z, the point we're being drawn towards (or away from)
    z_0 = np.random.normal(size=dim)
    # The variance of the noisy
    sigma_0 = np.random.uniform(0, 1)
    params = np.concatenate([alpha_0, w_0, z_0, [sigma_0]])
    
    return minimize(objective_func, params, args=(X, Y, alpha_dim, dim), bounds=[(-2, 2)] * alpha_dim + [(None, None)] * (2 * dim) + [(0, None)])['x']


def find_coeffs(X, Y, alpha_dim, dim): 
    # We'll run a linear regression
    reg = LinearRegression().fit(X, Y)
    return list(reg.coef_.flatten())


def run_sample(X, Y, smpl_size, iters, alpha_dim, dim):
    # Sample the data
    idxs = np.random.choice(range(len(X)), smpl_size)
    X_smpl = X[idxs]
    Y_smpl = X[idxs]
    return [find_coeffs(X_smpl, Y_smpl, alpha_dim, dim)]
    # return [run_opt(X_smpl, Y_smpl, alpha_dim, dim) for _ in range(iters)]


def run_experiment(expr_params):
    # Extract experimet parameters
    smpl_size = expr_params['smpl_size']
    iters = expr_params['iters']
    samps = expr_params['samps']
    dim = expr_params['dim']
    alpha_dim = expr_params['alpha_dim']

    # Fit over both sets of embeddings
    pca_trans = PCA(n_components=dim).fit(np.vstack([original_embeddings, improved_embeddings]))

    # X = PCA(n_components=dim).fit_transform(original_embeddings)
    # Y = PCA(n_components=dim).fit_transform(improved_embeddings)
    X = pca_trans.transform(original_embeddings)
    Y = pca_trans.transform(improved_embeddings)

    # Run the experiment
    res = []
    for _ in range(samps):
        res.extend(run_sample(X, Y, smpl_size, iters, alpha_dim, dim))
    return res

# res = find_coeffs(X_embedded, Y_embedded, 2, 2)
# print(res.shape)


# +
# Set up
expr_params = {
    'dim': 2,
    'alpha_dim': 1,
    'samps': 10,
    'smpl_size': 100,
    'iters': 100
}

# Run the experiment
res = run_experiment(expr_params)
print("Done")
# -

print(res)


# +
# Summary of results
def display_grad_results(res, expr_params):
    dim = expr_params['dim']
    alpha_dim = expr_params['alpha_dim']
    results_df = pd.DataFrame(res, columns=[f'alpha_{i}' for i in range(expr_params['alpha_dim'])] + [f'w_{i}' for i in range(dim)] + [f'z_{i}' for i in range(dim)] + ['sigma'])
    print(results_df.describe())

    # Plot the components
    components = ['alpha', 'w', 'z']  
    dims = [alpha_dim, dim, dim] 
    print(results_df.mean().values)
    # Three components:
    for i in range(3):
        xs = np.arange(dims[i])
        idxs = np.where(results_df.columns.str.contains(components[i]))[0]
        plt.figure()
        plt.bar(xs, results_df.mean().values[idxs], label='Mean')
        plt.title(components[i])
        plt.show()

def display_linear_results(res, expr_params):
    dim = expr_params['dim']
    results_df = pd.DataFrame(res, columns=[f'beta_{i}' for i in range(dim**2)])
    print(results_df.describe())


    xs = np.arange(dim**2)
    labs = [f'beta_{int(i/dim)}{i%dim}' for i in range(dim**2)]
    plt.figure()
    plt.bar(xs, results_df.mean().values, label='Mean')
    plt.xticks(xs, labs)
    plt.title('Coefficients')
    plt.show()

display_linear_results(res, expr_params)
# -

# Embeddings
pca_trans = PCA(n_components=2).fit(np.vstack([original_embeddings, improved_embeddings]))
X_embedded = pca_trans.transform(original_embeddings)
Y_embedded = pca_trans.transform(improved_embeddings)

# +
import matplotlib.pyplot as plt
ctr_pt = np.mean(results, axis=0)[alpha_dim + dim:alpha_dim + 2*dim]
print(ctr_pt)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], color='blue')
plt.scatter(Y_embedded[:, 0], Y_embedded[:, 1], color='red')
# plt.scatter(ctr_pt[0], ctr_pt[1], color='green')
plt.show()

# +
import seaborn as sns

# Kde plot of the results
# We need to convert to a dataframe

f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].set_title("Original")
sns.kdeplot(x = X_embedded[:, 0], y = X_embedded[:, 1], fill=True, alpha=0.4, ax = axs[0])
axs[1].set_title("Improved")
sns.kdeplot(x = Y_embedded[:, 0], y = Y_embedded[:, 1], fill=True, alpha=0.4, ax = axs[1])
# -


