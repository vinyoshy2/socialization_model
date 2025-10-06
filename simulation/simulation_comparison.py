from simulator import ModelSpecification
import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import json
import random 

def read3D(file_loc):
    output = []
    with open(file_loc) as f:
        dim1 = int(f.readline())
        dim2s = []
        for i in range(0, dim1):
            dim2s.append(int(f.readline()))
        dim3s = []
        for i in range(0, dim1):
            dim3s.append([])
            for j in range(0, dim2s[i]):
                dim3s[i].append(int(f.readline()))
            
        body = f.read()
        body = body.split()
        pos = 0
        for i in range(dim1):
            output.append([])
            for j in range(dim2s[i]):
                output[i].append([])
                for k in range(dim3s[i][j]):
                    output[i][j].append(float(body[pos]))
                    pos += 1
    return np.array(output)

def read_params(loc):
    with open(loc) as f:
        params = json.load(f)
    return params

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

def produce_samples_beta(iter_results):
    
    num_vectors = iter_results.shape[0]
    vector_size = iter_results.shape[1]
    num_iters = iter_results.shape[2]
    
    posterior = np.zeros((num_vectors, vector_size, num_iters))
    for cur_iter in range(0, num_iters):
        for cur_vector in range(0, num_vectors):
            params = iter_results[cur_vector,:, cur_iter]
            sampled_val = np.random.beta(a=params[1]+1, b=params[0]+1)
            posterior[cur_vector,0,cur_iter] = 1-sampled_val
            posterior[cur_vector,1,cur_iter] = sampled_val
    return posterior

def compare_spec(ground_truth, iter_results, name):
    num_vectors = iter_results.shape[0]
    vector_size = iter_results.shape[1]
    num_iters = iter_results.shape[2]
    if num_vectors > 10:
        sub_sample_vectors = random.sample(list(range(num_vectors)), k=10)
    else:
        sub_sample_vectors = list(range(num_vectors))
    print("num_vectors", num_vectors)
    print("vector_sizes", vector_size)
    print("num_iters", num_iters)
    print("sub_sample_inds", sub_sample_vectors)
    fig, axes = plt.subplots(nrows=len(sub_sample_vectors), ncols=1, figsize=(8, 20))
    for i, vec in enumerate(sub_sample_vectors):
        records = []
        for vec_ind in range(vector_size):
            records.append({"Index": vec_ind,
                            "Type": "Ground Truth",
                            "Probability": ground_truth[vec][vec_ind]})
            for iteration in range(num_iters):
                records.append({"Index": vec_ind,
                                            "Iter": iteration,
                                            "Type": "Inferred",
                                            "Probability": iter_results[vec][vec_ind][iteration]})
        cur_ax = axes[i]
        g = sns.pointplot(ax=cur_ax, data=pd.DataFrame.from_records(records),
                      x="Index", y="Probability", hue="Type", errorbar=("pi", 95), linestyles="")
        plt.setp(g.collections, alpha=0.5)
        cur_ax.set_ylim(0, 1)
        cur_ax.set_title("{} {}".format(name, vec))
    plt.savefig("{}/{}.pdf".format(plots_folder, name))
    return fig

params_folder = sys.argv[1]
model_output_folder = sys.argv[2]
plots_folder = sys.argv[3]
params = read_params("{}/params.json".format(params_folder))
#inferred_gammas = produce_samples_dirichlet(read3D("{}/gamma.txt".format(model_output_folder)), params["edge_list"])

#the ground truth values for gamma need special handling because the edge weight adjacency list is ragged
#gt_gammas = np.zeros((inferred_gammas.shape[0], inferred_gammas.shape[1]))
#for i in range(0, len(params["edge_weights"])):
#    for pos, j in enumerate(params["edge_list"][i]):
#        gt_gammas[i, j] = params["edge_weights"][i][pos]

#inferred_phis = produce_samples_dirichlet(read3D("{}/phi.txt".format(model_output_folder)))
#inferred_psis = produce_samples_dirichlet(read3D("{}/psi.txt".format(model_output_folder)))
#inferred_thetas = produce_samples_dirichlet(read3D("{}/theta.txt".format(model_ouptut_folder)))
lambda_pseudocounts = read3D("{}/lambda.txt".format(model_output_folder))
inferred_lambdas = produce_samples_beta(lambda_pseudocounts)
#the ground truth lambdas also need to be made into a 2D array
gt_lambdas = np.zeros((inferred_lambdas.shape[0], inferred_lambdas.shape[1]))
for i in range(0, len(params["coin_flip_probs"])):
    gt_lambdas[i][0] = 1 - params["coin_flip_probs"][i]
    gt_lambdas[i][1] = params["coin_flip_probs"][i]

#compare_spec(np.array(params["tgt_subreddit_topic_vectors"]), inferred_psis, "psis")
#compare_spec(gt_gammas, inferred_gammas, "gammas")
#compare_spec(np.array(params["vocab_vectors"]), inferred_phis, "phis")
compare_spec(gt_lambdas, inferred_lambdas, "Lambda")
#compare_spec(np.array(params["src_topic_vectors"]), inferred_thetas, "thetas")

