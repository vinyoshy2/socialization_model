import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import json
import sys
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


# Don't use this function long term, i think theres some incorrect approximinations (mean of means type stuff)
def read3D_sample_remap_space_efficient(file_loc, topk, idx2val=None):
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
    
        outer_dim = 0
        inner_dim = 0

        pos = 0
        output = []
        while outer_dim < dim1:
            cur_row = []
            while inner_dim < dim2s[outer_dim]:
                cur_line = f.readline()
                if cur_line.strip() != "":
                    nums = [int(val) for val in cur_line.split()]
                    if len(nums) <= 0:
                        print("~~~~{}~~~~".format(cur_line))
                    avg_num = np.mean(nums)
                    cur_row.append(avg_num)
                    inner_dim += 1
            row_norm = sum(cur_row)
            cur_row = [elem / row_norm for elem in cur_row]
            remapped_row = remap_vector(cur_row, topk, idx2val)
            output.append(remapped_row)
            outer_dim += 1
            inner_dim = 0
    return output


def readEdges(file_loc):
    edges = []
    with open(file_loc) as f:
        dim1 = int(f.readline())
        for i in range(dim1):
            f.readline()
        
        for i in range(dim1): 
            edge_line = f.readline()
            edge_indices = [int(x) for x in edge_line.split()]
            edges.append(edge_indices)

    return edges


def readJSON(file_loc):
    with open(file_loc) as f:
        data = json.load(f)
    return data


def produce_samples_beta(iter_results):
    num_subreddits = iter_results.shape[0]
    vector_size = iter_results.shape[1] 
    num_iters = iter_results.shape[2]
    
    posterior = np.zeros((num_subreddits, vector_size, num_iters))

    for cur_iter in range(0, num_iters):
        for cur_vector in range(0, num_subreddits):
            params = iter_results[cur_vector, :, cur_iter]
        
            sampled_val = np.random.beta(a=params[1]+1, b=params[0]+1)

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


def graph_lambda(all_records, in_path, out_path, verbose=False): 
    fig, axes = plt.subplots(figsize=(9, 6))

    idx2tgt_sub = readJSON(f"{in_path}/idx2tgt_sub.json")

    records = []
    for month, iter_results in all_records:
        num_lambdas = iter_results.shape[0]
        lambda_size = iter_results.shape[1]
        num_iters = iter_results.shape[2]

        for i, lambda_ in enumerate(range(num_lambdas)):
            for iteration in range(num_iters):
                records.append({"Subreddit": idx2tgt_sub[str(lambda_)],
                                "Month": month,
                                "Iter": iteration,
                                "Probability": iter_results[lambda_][0][iteration]})

    if verbose:
        df = pd.DataFrame.from_records(records)
        print("\nLambda Statistics:")
        print("=" * 80)
        for (month, subreddit), group in df.groupby(["Month", "Subreddit"]):
            mean_prob = group["Probability"].mean()
            hdi = az.hdi(group["Probability"].values, hdi_prob=0.95)
            print(f"Month {month}, {subreddit}: Mean = {mean_prob:.4f}, 95% HDI = [{hdi[0]:.4f}, {hdi[1]:.4f}]")
        print("=" * 80 + "\n")

    g = sns.lineplot(ax=axes, data=pd.DataFrame.from_records(records),
                x="Month", y="Probability", hue="Subreddit", marker='o', errorbar=("pi", 95), alpha=0.7)
    plt.setp(g.collections, alpha=0.5)
    axes.set_ylim(0, 1)
    axes.set_title("Lambda over Time")
    plt.savefig(f"{out_path}/Lambda.pdf")
    return fig


def compare_gamma(all_records, in_path, out_path, topk=-1):
    num_gammas = all_records.shape[0]
    gamma_size = all_records.shape[1]
    num_iters = all_records.shape[2]

    idx2tgt_pair = readJSON(f"{in_path}/idx2tgt_pair.json")
    idx2src_sub = readJSON(f"{in_path}/idx2src_sub.json")
    
    mean_gammas = np.mean(all_records, axis=2)
    
    with open(f"{out_path}/gamma_analysis.txt", "w") as f:
        # For each target document
        for target_idx in range(num_gammas):
            target_gamma = mean_gammas[target_idx]

            if topk > 0:
                 top_k_indices = np.argsort(target_gamma)[:topk][::-1]
                 top_k_values = target_gamma[top_k_indices]
            
            target_name = idx2tgt_pair[str(target_idx)]
            f.write(f"\nTarget Document {target_idx}: {target_name}\n")
            f.write("-" * 60 + "\n")
            f.write("Index | Source Subreddit | Gamma Value\n")
            f.write("-" * 60 + "\n")
            
            if topk > 0:
                for rank, (idx, val) in enumerate(zip(top_k_indices, top_k_values), 1):
                    source_name = idx2src_sub[str(idx)]
                    f.write(f"{rank:2d}   | {source_name:30s} | {val:.4f}\n")
            else:
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
    MODE = sys.argv[1]
    IN_PATH = sys.argv[2]
    OUT_PATH = sys.argv[3]

    # INPUTS_FOLDER = sys.argv[1]
    # RESULTS_FOLDER = sys.argv[2]
    # TOPICS = int(sys.argv[3])
    # START_MONTH = int(sys.argv[4])
    # END_MONTH = int(sys.argv[5])

    if MODE == "priors":
        from pathlib import Path

        PRIORS = [
            [0.1, 0.01, 0.1],
            [0.1, 1.0, 1.0],
            [1.0, 0.1, 0.01],
            [1.0, 1.0, 0.01],
            [0.01, 0.1, 0.01],
            [0.01, 1.0, 0.01],
            [0.01, 1.0, 0.1],
            [1.0, 0.1, 0.1],
            [1.0, 1.0, 0.1],
        ]

        for prior in PRIORS:
            print(f"\n========== alpha_topics {prior[0]}, alpha_vocab {prior[1]}, alpha_edges {prior[2]} ===========")
            if Path(f"{OUT_PATH}/alpha_topics_{prior[0]}_alpha_vocab_{prior[1]}_alpha_edges_{prior[2]}").is_dir():
                lambda_pseudocounts = read3D(f"{OUT_PATH}/alpha_topics_{prior[0]}_alpha_vocab_{prior[1]}_alpha_edges_{prior[2]}/lambda.txt")
                inferred_lambdas = produce_samples_beta(lambda_pseudocounts)
                graph_lambda([(2, inferred_lambdas)])


    if MODE == "lambda":
        START_MONTH = int(sys.argv[4])
        END_MONTH = int(sys.argv[5])

        all_records = []
        for MONTH in range(START_MONTH, END_MONTH + 1):
            lambda_pseudocounts = read3D(f"{RESULTS_FOLDER}/{MONTH}/lambda.txt")
            inferred_lambdas = produce_samples_beta(lambda_pseudocounts)
            all_records.append((MONTH, inferred_lambdas))
        graph_lambda(all_records)
        graph_lambda_by_iters(all_records)

    if MODE == "gamma":
        edges = readEdges(f"{IN_PATH}/edges.txt")
        gamma_pseudocounts = read3D(f"{OUT_PATH}/gamma.txt")
        inferred_gammas = produce_samples_dirichlet(gamma_pseudocounts, edges)
        compare_gamma(inferred_gammas)

    if MODE == "topics":
        MONTH = int(sys.argv[4])

        # read in files for converting indices to words/surbeddits
        idx2vocab = readJSON(f"{IN_PATH}/idx2vocab.json")
        idx2tgt_pair = readJSON(f"{IN_PATH}/idx2tgt_pair.json")
        idx2tgt_sub = readJSON(f"{IN_PATH}/idx2tgt_sub.json")
        idx2src_sub = readJSON(f"{IN_PATH}/idx2src_sub.json")
        print("read in jsons")

        #read in and compute posteriors
        phi = read3D_sample_remap_space_efficient(f"{OUT_PATH}/{MONTH}/phi.txt", 40, idx2val=idx2vocab)
        display_topics(phi)
        psi = read3D_sample_remap_space_efficient(f"{OUT_PATH}/{MONTH}/psi.txt", 10, idx2val=None)
        display_document_preprocessed(psi, phi, idx2tgt_sub)
        theta = read3D_sample_remap_space_efficient(f"{OUT_PATH}/theta.txt", 10, idx2val=None)
        display_document_preprocessed(theta, phi, idx2src_sub)




    
#
