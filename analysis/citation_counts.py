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
OUTPUTS_FOLDER = sys.argv[3]


#don't use this function long term, i think theres some incorrect approximinations (mean of means type stuff)
def read_and_process_c(file_loc, row_func, post_processed_shape):
    output = []
    with open(file_loc) as f:
        # Read first dimension (number of iterations -- MCMC samples)
        dim1 = int(f.readline())
        # Read second dimension (number of comments)
        dim2s = []
        for i in range(0, dim1):
            dim2s.append(int(f.readline()))
        # Read third dimension  (number of words per comment)
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
            cur_rows = []
            while inner_dim < dim2s[outer_dim]:
                cur_line = f.readline()
                if cur_line.strip() != "":
                    nums = [int(val) for val in cur_line.split()]
                    nums = row_func(nums, inner_dim)
                    cur_rows.append(nums)
                    inner_dim += 1
            output.append(cur_rows)
            outer_dim += 1
            inner_dim = 0
        avg = [np.zeros(post_processed_shape) for i in range(dim2s[0])]
        for j in dim2:
            for i in range(0, dim1):
                avg[j] += np.array(output[i][j])
            avg[j] = avg[j] / dim1
    return avg

def cited_this_iteration(iter_vec, comment_number, num_possible_cites):
    vals, counts = np.unique(iter_vec, return_counts=True)
    output = np.zeros(num_possible_cites)
    for i, elem in enumerate(vals):
        output[elem] = counts[i]
    return output

def words_by_source(iter_vec, comment_number, num_possible_cites, vocab_size, lookup):
    output = np.zeros((num_possible_cites, vocab_size))
    position in range(iter_vec.shape[0]):
        cited_sub = iter_vec[pos][iter_num]
        word = int(lookup[doc_number][pos])
        output[cited_sub][word] += 1
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

def read1D(file_loc):
    output = []
    with open(file_loc) as f:
        # Read first dimension (number of subreddits)
        dim1 = int(f.readline()) # = 2 (coronavirus, china_flu)
        # Read the rest of file
        body = f.read()
        # Split into individual numbers
        body = body.split()
        # Populate output 3D list
        for i in range(dim1): # For each subreddit
            output.append(float(body[i]))
    return output

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

def top_k_cite(row, k, idx2str):
    indices = list(range(len(row))
    indices = sorted(indices, key = lambda x: row[x], reverse=True)
    pos = 0
    ret_val = {}
    while pos < k:
        ret_val[idx2str[pos]] = row[pos]
        pos += 1
    return ret_val

def top_k_word_cite(row, k, idx2src_sub, idx2vocab):
    subreddit_indices = list(range(row.shape[0]))
    word_indices = list(range(row.shape[1]))
    subreddit_indices = sorted(indices, key = lambda x: sum(row[x]), reverse=True)
    sub_pos = 0
    all_subs = {}
    while sub_pos < k:
        cur_sub_idx = subreddit_indices[sub_pos]
        subreddit= idx2src_sub[cur_sub_idx]
        all_subs[subreddit] = {}
        word_pos = 0
        word_indices = sorted(word_indices, key = lambda x: row[cur_sub_idx][x], reverse=True)
        while word_pos < k:
            cur_word_idx = word_indices[word_pos]
            cur_word_val = row[cur_sub_idx][cur_word_idx]
            cur_word = idx2vocab[cur_word_idx]
            all_subs[subreddit][cur_word] = cur_word_val
            word_pos += 1
        sub_pos += 1
    return all_subs


def aggregate_and_top_k(citation_counts, k, vector_shape, top_k_func, subreddits, edges, idx2tgt_subs):
    num_comments = len(citation_counts)
    all_subs = list(set(subreddits))
    results = {}
    comment_counts = {}
    for sub in all_subs:
        results[sub] = np.zeros(vector_shape)
        comment_counts[sub] = 0
    for i in range(num_comments):
        cur_counts = citation_counts[i]
        cur_subreddit = subreddits[i]
        cur_edges = edges[i]
        cur_tgt_sub = idx2tgt_sub[cur_subreddit]
        #only include comments who had the choice to cite ?
        if len(cur_edges) > 0:
            results[sub] += cur_counts
            comment_counts[sub] += 1
    for sub in all_subs:
        results[sub] = results[sub] / comment_counts[sub]
        results[sub] = top_k_func(row, k)
    return results


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
    edges = readEdges("{}/edges.txt".format(INPUTS_FOLDER))
    subreddits = read1D("{}/subreddits.txt".format(INPUTS_FOLDER))

    #process files
    citation_counts = read_and_process_c("{}/assign_c.txt".format(INPUTS_FOLDER),
                                         lambda x, y: cited_this_iteration(x, y, len(idx2src_sub)),
                                         (len(idx2src_sub),)
                      )
    word_citation_counts = read_and_process_c("{}/assign_c.txt".format(INPUTS_FOLDER),
                                              lambda x, y: words_by_source(x, y, len(idx2src_sub), len(idx2vocab)),
                                              (len(idx2src_sub), len(idx2vocab)),
                                              tgt_blobs
                           )
   

    #TO-DO: Pull top-k cites per doc, and top-k words per cite per doc
    cites_per_doc = aggregate_and_top_k(citation_counts,
                                        20,
                                        (len(idx2src_sub), ),
                                        lambda x, y: top_k_cite(x, y, idx2src_sub),
                                        subreddits,
                                        edges,
                                        idx2tgt_subs):
    words_per_cite_per_doc = top_k_words_per_cite(word_citations_counts,
                                                  20,
                                                  (len(idx2src_sub), len(idx2vocab)),
                                                  lambda x, y: top_k_word_cite(x, y, idx2src_sub, idx2vocab)
                                                  subreddits,
                                                  edges,
                                                  idx2tgt_sub
                             )
    with open("{}/top_k_cites.json".format(OUTPUTS_FOLDER), "w+") as f:
        f.write(json.dumps(cites_per_doc, indent=4))
    with open("{}/top_k_words_per_cite.json".format(OUTPUTS_FOLDER), "w+") as f:
        f.write(json.dumps(words_per_cite_per_doc, indent=4))
