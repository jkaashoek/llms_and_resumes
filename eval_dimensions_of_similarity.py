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
from text_helpers import TextPool
import numpy as np 
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# -

profiles = pd.read_csv("resumes/profiles_with_updates.csv")
profiles.head()

original_embeddings = embedder.encode(profiles['original'], show_progress_bar=True)
updated_embeddings = embedder.encode(profiles['improve'], show_progress_bar=True)
print("Done")

print(cosine_similarity(original_embeddings))


# +

def bootstrap(data, eval_metrics, iters=100, samp_size=100):
    if type(eval_metrics) != list:
        eval_metrics = [eval_metrics]
    
    true_evals = [m(data) for m in eval_metrics]
    bootstrapped_evals = np.empty((iters, len(eval_metrics)))
    n, d = data.shape
    idxs = np.arange(n)

    for i in range(iters):
        samp_idxs = np.random.choice(idxs, size=samp_size, replace=True)
        sample = data[samp_idxs]
        evals = [m(sample) for m in eval_metrics]
        bootstrapped_evals[i] = evals

    return true_evals, bootstrapped_evals

def med_sim(data):
    # cos similarity between each row and all other rows
    sims = cosine_similarity(data)
    # remove diagonal
    sims = sims[~np.eye(sims.shape[0],dtype=bool)].reshape(data.shape[0],-1)
    return np.median(sims)

def unique_words(data):
    return len(set(data.flatten()))

def report_bootstrap(true_evals, bootstrapped_evals, metric_names):
    for i, true_eval in enumerate(true_evals):
        boot_evals = bootstrapped_evals[:, i]
        ci = np.percentile(boot_evals, [2.5, 97.5])
        print(f"{metric_names[i]}: {true_eval} (95% CI: {ci[0]:.2f} - {ci[1]:.2f})")

metric_names = ["Median similarity", "Unique words"]
sim, boot_sim = bootstrap(original_embeddings, [med_sim, unique_words], iters=1000, samp_size=100)
print("original text")
report_bootstrap(sim, boot_sim, metric_names)

sim, boot_sim = bootstrap(updated_embeddings, [med_sim, unique_words], iters=1000, samp_size=100)
print("Improved text")
report_bootstrap(sim, boot_sim, metric_names)
