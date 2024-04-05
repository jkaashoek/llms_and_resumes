# from consts import embedding_model
import time
import os
from sentence_transformers import SentenceTransformer

# How well does this work on texts?

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

text_dir = 'resumes/kaggle_small/'

# Read each into a list
texts = []
for f in os.listdir(text_dir):
    with open(f'{text_dir}/{f}', 'r') as file:
        texts.append(file.read())

print(len(texts))

def time_to_encode(texts):
    start_time = time.time()
    _ = embedding_model.encode(texts)
    end_time = time.time()
    return end_time - start_time

def time_to_encode_one_at_a_time(texts):
    start_time = time.time()
    for t in texts:
        _ = embedding_model.encode(t)
    end_time = time.time()
    return end_time - start_time


# Let's do a few iteration of each
n = 5
print("running tests... all at once")
times = [time_to_encode(texts) for _ in range(n)]
print("running tests... one at a time")
times_one_at_a_time = [time_to_encode_one_at_a_time(texts) for _ in range(n)]

print(f"Average time to encode all at once: {sum(times) / n}")
print(f"Average time to encode one at a time: {sum(times_one_at_a_time) / n}")