import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import json
import random 

MONTHS = int(sys.argv[1])
TOPICS = int(sys.argv[2])

RESULTS_FOLDER = f"data/results/topics-{TOPICS}"
SUBREDDIT_NAMES = {0: "coronavirus", 1: "china_flu"}

def read3D(file_loc):
    output = []
    with open(file_loc) as f:
        # Read first dimension (number of subreddits)
        dim1 = int(f.readline()) # = 2 (coronavirus, china_flu)

        # Read second dimension (components per subreddit; probability pairs)
        dim2s = []
        for i in range(0, dim1):
            dim2s.append(int(f.readline())) # = [2, 2]

        # Read third dimension  (iterations per component; MCMC samples)
        dim3s = []
        for i in range(0, dim1):
            dim3s.append([])
            for j in range(0, dim2s[i]):
                dim3s[i].append(int(f.readline())) # = [[1500, 1500], [1500, 1500]]
        
        # Read the rest of file
        body = f.read()
        # Split into individual numbers
        body = body.split()

        # Populate output 3D list
        pos = 0
        for i in range(dim1): # For each subreddit
            output.append([])
            for j in range(dim2s[i]): # For each probability pair
                output[i].append([])
                for k in range(dim3s[i][j]): # For each iteration
                    output[i][j].append(float(body[pos]))
                    pos += 1

    return np.array(output)

def readEdges(file_loc):
    edges = []
    with open(file_loc) as f:
        # Read first dimension (number of target pairs)
        dim1 = int(f.readline())

        # Skip through the edge counts
        for i in range(dim1):  # For each target pair
            f.readline()
        
        # Read edges
        for i in range(dim1):  # For each target pair
            edge_line = f.readline()
            edge_indices = [int(x) for x in edge_line.split()]
            edges.append(edge_indices)

    return edges

def produce_samples_beta(iter_results):
    # Get dimensions from input array
    num_subreddits = iter_results.shape[0]  # 2 subreddits
    vector_size = iter_results.shape[1]     # 2 components (p, 1-p)
    num_iters = iter_results.shape[2]       # 1500 MCMC iterations
    
    # Initialize output array with same shape
    posterior = np.zeros((num_subreddits, vector_size, num_iters))

    # For each MCMC iteration
    for cur_iter in range(0, num_iters):
        # For each subreddit
        for cur_vector in range(0, num_subreddits):
            # Get pseudocounts for this iteration and subreddit
            params = iter_results[cur_vector,:, cur_iter] # [count0, count1]
            
            # Sample from Beta distribution
            sampled_val = np.random.beta(a=params[1]+1, b=params[0]+1)

            # Store probability pair (1-p, p)
            posterior[cur_vector,0,cur_iter] = 1-sampled_val
            posterior[cur_vector,1,cur_iter] = sampled_val
            
    return posterior


def produce_samples_dirichlet(iter_results, non_zero_list = None):
    num_vectors = iter_results.shape[0]
    vector_size = iter_results.shape[1]
    num_iters = iter_results.shape[2]
    
    posterior = np.zeros((num_vectors, vector_size, num_iters))
    
    for cur_iter in range(0, num_iters):
        for cur_vector in range(0, num_vectors):
            orig_params = iter_results[cur_vector,:, cur_iter]
            if non_zero_list != None:
                params = [orig_params[i] for i in non_zero_list[cur_vector]]
            else:
                params = orig_params
            print(cur_vector, non_zero_list[cur_vector])
            print(orig_params, np.count_nonzero(orig_params))
            print(params)
            sampled_vector = np.random.dirichlet(params)

            if non_zero_list != None:
                reformed_sampled = np.zeros(vector_size)
                for val, index in zip(sampled_vector, non_zero_list[cur_vector]):
                    reformed_sampled[index] = val
                sampled_vector = reformed_sampled
            posterior[cur_vector,:,cur_iter] = sampled_vector
    return posterior


def graph_lambda(all_records): 
    fig, axes = plt.subplots(figsize=(9, 6))
    records = []
    for month, iter_results in all_records:
        num_vectors = iter_results.shape[0]
        vector_size = iter_results.shape[1]
        num_iters = iter_results.shape[2]

        sub_sample_vectors = list(range(num_vectors))
        for i, vec in enumerate(sub_sample_vectors):
            for vec_ind in range(vector_size):
                for iteration in range(num_iters):
                    records.append({"Subreddit": SUBREDDIT_NAMES[vec_ind],
                                    "Month": month,
                                    "Iter": iteration,
                                    "Probability": iter_results[vec][0][iteration]})

    g = sns.lineplot(ax=axes, data=pd.DataFrame.from_records(records),
                x="Month", y="Probability", hue="Subreddit", marker='o', errorbar=("pi", 95), alpha=0.7)
    plt.setp(g.collections, alpha=0.5)
    axes.set_ylim(0, 1)
    axes.set_title("Lambda over Time")
    plt.savefig(f"{RESULTS_FOLDER}/Lambda.pdf")
    return fig


def compare_gamma(all_records):
    num_gammaas = all_records.shape[0]  # Target Documents: tgt_pair (idx2tgt_pair)
    gamma_size = all_records.shape[1]   # Source Documents: src_sub (idx2src_sub)
    num_iters = all_records.shape[2]    # 1500 MCMC Iterations
    
    with open("data/1/idx2tgt_pair.json", "r") as f:
        idx2tgt_pair = json.load(f)
    with open("data/1/idx2src_sub.json", "r") as f:
        idx2src_sub = json.load(f)
    
    mean_gammas = np.mean(all_records, axis=2)
    
    with open(f"{RESULTS_FOLDER}/gamma_analysis-0.txt", "w") as f:
        # For each target document
        # for target_idx in range(num_gammaas):
        target_idx = 0
        target_gamma = mean_gammas[target_idx]
        
        # top_5_indices = np.argsort(target_gamma)[:][::-1]
        # top_5_values = target_gamma[top_5_indices]
        
        target_name = idx2tgt_pair[str(target_idx)]
        f.write(f"\nTarget Document {target_idx}: {target_name}\n")
        f.write("-" * 60 + "\n")
        f.write("Index | Source Subreddit | Gamma Value\n")
        f.write("-" * 60 + "\n")
        
        # for rank, (idx, val) in enumerate(zip(top_5_indices, top_5_values), 1):
        #     source_name = idx2src_sub[str(idx)]
        #     f.write(f"{rank:2d}   | {source_name:30s} | {val:.4f}\n")
        for idx, val in enumerate(target_gamma, 1):
            source_name = idx2src_sub[str(idx)]
            f.write(f"{idx:2d}   | {source_name:30s} | {val:.4f}\n")


if __name__ == "__main__":
    ### LAMBDA ###
    # all_records = []
    # for month in range(1, MONTHS + 1):
    #     lambda_pseudocounts = read3D(f"{RESULTS_FOLDER}/{month}/lambda.txt")
    #     inferred_lambdas = produce_samples_beta(lambda_pseudocounts)
    #     all_records.append((month, inferred_lambdas))
    # graph_lambda(all_records)

    ### GAMMA ###
    edges = readEdges(f"data/1/edges.txt")
    gamma_pseudocounts = read3D(f"{RESULTS_FOLDER}/1/gamma.txt")
    inferred_gammas = produce_samples_dirichlet(gamma_pseudocounts, edges)
    # compare_gamma(inferred_gammas)
    
