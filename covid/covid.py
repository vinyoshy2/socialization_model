import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import json
import random 

#TOPICS = int(sys.argv[2])

INPUTS_FOLDER = sys.argv[1]
RESULTS_FOLDER = sys.argv[2]
MONTH = int(sys.argv[3])
RESULTS_FOLDER = "{}/{}/".format(RESULTS_FOLDER, MONTH)
INPUTS_FOLDER = "{}/{}/".format(INPUTS_FOLDER, MONTH)
SUBREDDIT_NAMES = {0: "coronavirus", 1: "china_flu"}


#don't use this function long term, i think theres some incorrect approximinations (mean of means type stuff)
def analyze_assign_c(file_loc, topk, comments_per_sub, vector_shape, processing_func, lookup):
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
    
        outer_dim = 0
        inner_dim = 0
        # Populate output 3D list
        pos = 0
        output = []
        cur_row_avg = np.zeros(vector_shape)
        while outer_dim < dim1:
            cur_row = []
            while inner_dim < dim2s[outer_dim]:
                cur_line = f.readline()
                if cur_line.strip() != "":
                    nums = [int(val) for val in cur_line.split()]
                    if len(nums) <= 0:
                        print("~~~~{}~~~~".format(cur_line))
                    cur_row.append(nums)
                    inner_dim += 1
            cur_row_avg += processing_func(cur_row, comments_per_sub, vector_shape, outer_dim, lookup)
            if (outer_dim % comments_per_sub) == (comments_per_sub - 1):
                remapped_row = cur_row_avg
                output.append(remapped_row)
                cur_row_avg = np.zeros(vector_shape)
            outer_dim += 1
            inner_dim = 0
    return output

def cited_this_iteration(iter_vec, comments_per_sub, vector_shape, doc_number, lookup=None):
    filtered = []
    for row in iter_vec:
        if not all([val == 0 for val in row]):
            filtered.append(row)
    vals, counts = np.unique(filtered, return_counts=True)
    output = np.zeros(vector_shape)
    for i, elem in enumerate(vals):
        output[elem] = counts[i]
    return output

def words_by_source(iter_vec, comments_per_sub, vector_shape, doc_number, lookup):
    filtered = []
    for row in iter_vec:
        if not all([val == 0 for val in row]):
            filtered.append(row)
    output = np.zeros(vector_shape)
    iter_vec = np.array(filtered)
    if len(iter_vec.shape) > 1:
        for iter_num in range(iter_vec.shape[1]):
            for pos in range(iter_vec.shape[0]):
                cited_sub = iter_vec[pos][iter_num]
                word = int(lookup[doc_number][pos])
                output[cited_sub][word] += 1
    return output

#don't use this function long term, i think theres some incorrect approximinations (mean of means type stuff)
def read3Dgamma_sample_remap_space_efficient(file_loc, topk, comments_per_sub, idx2val=None):
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
    
        outer_dim = 0
        inner_dim = 0
        # Populate output 3D list
        pos = 0
        output = []
        cur_row_avg = np.zeros(dim2s[0])
        while outer_dim < dim1:
            cur_row = []
            while inner_dim < dim2s[outer_dim]:
                cur_line = f.readline()
                if cur_line.strip() != "":
                    nums = [float(val) for val in cur_line.split()]
                    if len(nums) <= 0:
                        print("~~~~{}~~~~".format(cur_line))
                    avg_num = np.mean(nums)
                    cur_row.append(avg_num)
                    inner_dim += 1
            row_norm = sum(cur_row)
            if row_norm != 0:
                cur_row = [elem / row_norm for elem in cur_row]
            if (outer_dim % comments_per_sub) != (comments_per_sub - 1):
                cur_row_avg += np.array(cur_row)
            else:
                remapped_row = remap_vector(cur_row_avg / comments_per_sub, topk, idx2val)
                output.append(remapped_row)
                cur_row_avg = np.zeros(dim2s[0])
            outer_dim += 1
            inner_dim = 0
    return output
#don't use this function long term, i think theres some incorrect approximinations (mean of means type stuff)
def read3D_sample_remap_space_efficient(file_loc, topk, idx2val=None):
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
    
        outer_dim = 0
        inner_dim = 0
        # Populate output 3D list
        pos = 0
        output = []
        while outer_dim < dim1:
            cur_row = []
            while inner_dim < dim2s[outer_dim]:
                cur_line = f.readline()
                if cur_line.strip() != "":
                    nums = [float(val) for val in cur_line.split()]
                    if len(nums) <= 0:
                        print("~~~~{}~~~~".format(cur_line))
                    avg_num = np.mean(nums)
                    cur_row.append(avg_num)
                    inner_dim += 1
            row_norm = sum(cur_row)
            if row_norm != 0:
                cur_row = [elem / row_norm for elem in cur_row]
            remapped_row = remap_vector(cur_row, topk, idx2val)
            output.append(remapped_row)
            outer_dim += 1
            inner_dim = 0
    return output

def read2D(file_loc):
    output = []
    with open(file_loc) as f:
        # Read first dimension (number of subreddits)
        dim1 = int(f.readline()) # = 2 (coronavirus, china_flu)

        # Read second dimension (components per subreddit; probability pairs)
        dim2s = []
        for i in range(0, dim1):
            dim2s.append(int(f.readline())) # = [2, 2]

        # Read the rest of file
        body = f.read()
        # Split into individual numbers
        body = body.split()

        # Populate output 3D list
        pos = 0
        for i in range(dim1): # For each subreddit
            output.append([])
            for j in range(dim2s[i]): # For each probability pair
                output[i].append(float(body[pos]))
                pos += 1

    return output

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

def readJSON(file_loc):
    with open(file_loc) as f:
        data = json.load(f)
    return data

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


def remap_vector(vector, topk, idx2vocab=None):
    if idx2vocab != None:
        pairs = [(idx2vocab[str(j)], value) for j, value in enumerate(vector)]
    else:
        pairs = [(j, value) for j, value in enumerate(vector)]
    pairs.sort(key = lambda x: x[1], reverse=True)
    return pairs[:topk]

def display_topic(topic_pairs):
    row_str = " | ".join(["{} ({})".format(pair[0], pair[1]) for pair in topic_pairs])
    print(row_str)

def display_topics(topic_vectors):
    for i, topic_vector in enumerate(topic_vectors):
        print("--- Topic {} ---".format(i))
        display_topic(topic_vector)



def display_document_preprocessed(documents, vocab_vectors, idx2subreddit):
    for i in range(len(documents)):
        cur_sub = idx2subreddit[str(i)]
        doc_row = documents[i]
        print(doc_row)
        print("================= {} =================".format(cur_sub))
        for k in range(len(doc_row)):
            cur_topic = doc_row[k][0]
            cur_topic_prop = doc_row[k][1]
            print("--- Topic {} ({}) ---".format(cur_topic, cur_topic_prop))
            display_topic(vocab_vectors[cur_topic])

if __name__ == "__main__":
    #### LAMBDA ###
#    all_records = []
#    lambda_pseudocounts = read3D(f"{RESULTS_FOLDER}/lambda.txt")
#    inferred_lambdas = produce_samples_beta(lambda_pseudocounts)
#    all_records.append((MONTH, inferred_lambdas))
#    graph_lambda(all_records)

    
    ### TOPIC VECTORS  ###
    
    #read in files for converting indices to words/surbeddits
    idx2vocab = readJSON("{}/idx2vocab.json".format(INPUTS_FOLDER))
    idx2tgt_pair = readJSON("{}/idx2tgt_pair.json".format(INPUTS_FOLDER))
    idx2tgt_sub = readJSON("{}/idx2tgt_sub.json".format(INPUTS_FOLDER))
    idx2src_sub = readJSON("{}/idx2src_sub.json".format(INPUTS_FOLDER))
    idx2src_sub[str(len(idx2src_sub))] = "self"
    tgt_blobs = read2D("{}/tgt_blobs.txt".format(INPUTS_FOLDER))
    
    #output = analyze_assign_c("{}/assign_c.txt".format(RESULTS_FOLDER), 10, 100, len(idx2src_sub), cited_this_iteration, None)
    #remapped = [[],[]]
    #remapped[0] = remap_vector(output[0], 20, idx2src_sub)
#    remapped[1] = remap_vector(output[1], 20, idx2src_sub)
#   display_topics(remapped)
 #   print("dim1", output)
  #  print("dim1", len(output))
   # print("dim2", len(output[0]))
    ### GAMMA ###
 #   edges = readEdges("{}/edges.txt".format(INPUTS_FOLDER))
  #  gammas = read3Dgamma_sample_remap_space_efficient("{}/gamma.txt".format(RESULTS_FOLDER), 10, 100, idx2val=idx2src_sub)
   # display_topics(gammas)
    #inferred_gammas = produce_samples_dirichlet(gamma_pseudocounts, edges)
    # compare_gamma(inferred_gammas)

    #read in and compute posteriors
   # phi = read3D_sample_remap_space_efficient("{}/phi.txt".format(RESULTS_FOLDER), 40, idx2val=idx2vocab)
   # display_topics(phi)
    #psi = read3D_sample_remap_space_efficient("{}/psi.txt".format(RESULTS_FOLDER), 10, idx2val=None)
    #theta = read3D_sample_remap_space_efficient("{}/theta.txt".format(RESULTS_FOLDER), 10, idx2val=None)
    #display_document_preprocessed(psi, phi, idx2tgt_sub)
    #display_document_preprocessed(theta, phi, idx2src_sub)


    output = analyze_assign_c("{}/assign_c.txt".format(RESULTS_FOLDER), 10, 100, (len(idx2src_sub), len(idx2vocab)), words_by_source, tgt_blobs)
    for i, row in enumerate(output[0]):
        if sum(row) > 1000:
            print("========== {} =========".format(idx2src_sub[str(i)]))
            remapped = remap_vector(row, 20, idx2vocab)
            display_topic(remapped)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i, row in enumerate(output[1]):
        if sum(row) > 1000:
            print("========== {} =========".format(idx2src_sub[str(i)]))
            remapped = remap_vector(row, 20, idx2vocab)
            display_topic(remapped)
            


    

