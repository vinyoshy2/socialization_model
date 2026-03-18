import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import json
import os
import random 
import time
#TOPICS = int(sys.argv[2])

INPUTS_FOLDER = sys.argv[1]
RESULTS_FOLDER = sys.argv[2]
OUTPUTS_FOLDER = sys.argv[3]

def filter_and_remap(cites_per_doc, src_sub2idx):
    old2new = {}
    new2old = {}
    new_idx = 0
    for tgt_sub in cites_per_doc:
        for src_sub in cites_per_doc[tgt_sub].keys():
            old_idx = src_sub2idx[src_sub]
            if old_idx not in old2new:
                old2new[old_idx] = new_idx
                new2old[new_idx] = old_idx
                new_idx += 1
    return old2new, new2old

#don't use this function long term, i think theres some incorrect approximinations (mean of means type stuff)
def read_and_process_c(file_loc, row_func, post_processed_shape, edges, subreddits, lookup=None):
    output = []
    num_subreddits = max(subreddits) + 1
    with open(file_loc) as f:
        # Read first dimension (number of iterations -- MCMC samples)
        dim1 = int(f.readline())
        # Read second dimension (number of comments)
        dim2s = []
        for i in range(0, dim1):
            dim2 = int(f.readline())
            dim2s.append(dim2)
        # Read third dimension  (number of words per comment)
        dim3s = []
        for i in range(0, dim1):
            dim3s.append([])
            for j in range(0, dim2s[i]):
                dim3 = int(f.readline())
                dim3s[i].append(dim3) # = [[1500, 1500], [1500, 1500]]
        outer_dim = 0
        inner_dim = 0
        # Populate output 3D list
        pos = 0
        avg = [np.zeros(post_processed_shape) for i in range(num_subreddits)]
        words_per_sub = {sub: 0 for sub in range(num_subreddits)}
        while outer_dim < dim1:
            start = time.time()
            if outer_dim % 100 == 0:
                print("{}, iter {}".format(file_loc, outer_dim))
            while inner_dim < dim2s[outer_dim]:
                start = time.time()
                subreddit = subreddits[inner_dim]
                cur_line = f.readline()
                #skip comment's that are forced innovations
                if len(edges[inner_dim]) > 0:
                    if cur_line.strip() != "":
                        nums = [int(val) for val in cur_line.split()]
                        nums = row_func(nums, inner_dim, lookup)
                    else:
                        nums = np.zeros(post_processed_shape)
                    avg[subreddit] += nums
                    words_per_sub[subreddit] += nums.sum()
                inner_dim += 1
            f.readline()
            f.readline()
            outer_dim += 1
            inner_dim = 0
        for j in range(num_subreddits):
            avg[j] = avg[j] / (words_per_sub[j])
    return avg

def cited_this_iteration(iter_vec, comment_number, num_possible_cites):
    vals, counts = np.unique(iter_vec, return_counts=True)
    output = np.zeros(num_possible_cites)
    for i, elem in enumerate(vals):
        output[elem] = counts[i]
    return output

def words_by_source_vectorized(iter_vec, comment_number, num_possible_cites, vocab_size, idx_map, lookup):
    print(len(iter_vec))
    print(len(lookup[comment_number]))
    output = np.zeros((num_possible_cites, vocab_size))
    
    iter_arr = np.array(iter_vec)
    words = np.array(lookup[comment_number], dtype=int)
    
    # Find positions where iter_vec values exist in idx_map
    mapped = np.array([idx_map.get(v, -1) for v in iter_arr])
    valid_mask = mapped != -1
    
    cited_indices = mapped[valid_mask]
    word_indices = words[valid_mask]
    
    np.add.at(output, (cited_indices, word_indices), 1)
    
    return output

def words_by_source(iter_vec, comment_number, num_possible_cites, vocab_size, idx_map, lookup):
    print(len(iter_vec))
    print(len(lookup[comment_number]))
    output = np.zeros((num_possible_cites, vocab_size))
    for position in range(len(iter_vec)):
        cited_sub = iter_vec[position]
        if cited_sub in idx_map:
            cited_sub = idx_map[cited_sub]    
            word = int(lookup[comment_number][position])
            output[cited_sub][word] += 1
    return output

def write2D(file_loc, data):
    with open(file_loc, 'w') as f:
        dim1 = len(data)
        f.write(f"{dim1}\n")
        for i in range(dim1):
            f.write(f"{len(data[i])}\n")
        for i in range(dim1):
            f.write(" ".join(str(data[i][j]) for j in range(len(data[i]))))
            f.write("\n\n")


def write3D(file_loc, data):
    with open(file_loc, 'w') as f:
        dim1 = len(data)
        f.write(f"{dim1}\n")
        for i in range(dim1):
            f.write(f"{len(data[i])}\n")
        for i in range(dim1):
            for j in range(len(data[i])):
                f.write(f"{len(data[i][j])}\n")
        for i in range(dim1):
            for j in range(len(data[i])):
                f.write(" ".join(str(data[i][j][k]) for k in range(len(data[i][j]))))
                f.write("\n")
            f.write("\n")


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

        # Populate output 2D list
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
        # Read first dimension
        dim1 = int(f.readline())
        # Read second dimensions
        dim2s = []
        for i in range(dim1):
            dim2s.append(int(f.readline()))
        # Read third dimensions
        dim3s = []
        for i in range(dim1):
            for j in range(dim2s[i]):
                dim3s.append(int(f.readline()))
        # Read the rest of file
        body = f.read().split()
        # Populate output 3D list
        pos = 0
        dim3_idx = 0
        for i in range(dim1):
            output.append([])
            for j in range(dim2s[i]):
                output[i].append([])
                for k in range(dim3s[dim3_idx]):
                    output[i][j].append(float(body[pos]))
                    pos += 1
                dim3_idx += 1
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
    indices = list(range(len(row)))
    indices = sorted(indices, key = lambda x: row[x], reverse = True)
    pos = 0
    ret_val = {}
    while pos < k:
        index = indices[pos]
        ret_val[idx2str[str(index)]] = row[index]
        pos += 1
    return ret_val

def top_k_word_cite(row, k, idx2src_sub, idx2vocab, tgt_blobs, idx_remap):
    word_indices = list(range(len(row[0])))
    subreddit_indices = list(range(len(row)))
    subreddit_indices = sorted(subreddit_indices, key=lambda x: sum(row[x]), reverse=True)
    all_subs = {}
    for cur_sub_idx in subreddit_indices[:k]:
        subreddit= idx2src_sub[str(idx_remap[cur_sub_idx])]
        all_subs[subreddit] = {}
        word_indices = sorted(word_indices, key = lambda x: row[cur_sub_idx][x], reverse=True)
        for cur_word_idx in word_indices[:k]:
            cur_word_val = row[cur_sub_idx][cur_word_idx]
            cur_word = idx2vocab[str(cur_word_idx)]
            all_subs[subreddit][cur_word] = cur_word_val
    return all_subs


def aggregate_and_top_k(citation_counts, k, vector_shape, top_k_func, subreddits, idx2tgt_sub):
    num_subs = len(citation_counts)
    all_subs = list(set(idx2tgt_sub.values()))
    results = {}
    for i in range(num_subs):
        cur_counts = citation_counts[i]
        cur_subreddit = idx2tgt_sub[str(i)]
        results[cur_subreddit] = top_k_func(cur_counts, k)
    return results

if __name__ == "__main__":
    
    #read in files for converting indices to words/surbeddits
    idx2vocab = readJSON("{}/idx2vocab.json".format(INPUTS_FOLDER))
    idx2tgt_pair = readJSON("{}/idx2tgt_pair.json".format(INPUTS_FOLDER))
    idx2tgt_sub = readJSON("{}/idx2tgt_sub.json".format(INPUTS_FOLDER))
    idx2src_sub = readJSON("{}/idx2src_sub.json".format(INPUTS_FOLDER))
    src_sub2idx = readJSON("{}/src_sub2idx.json".format(INPUTS_FOLDER))
    idx2src_sub[str(len(idx2src_sub))] = "self"
    tgt_blobs = read2D("{}/tgt_blobs.txt".format(INPUTS_FOLDER))
    edges = readEdges("{}/edges.txt".format(INPUTS_FOLDER))
    subreddits = read1D("{}/subreddits.txt".format(INPUTS_FOLDER))
    subreddits = [int(subreddit) for subreddit in subreddits]

    #process files
    citation_counts_file = "{}/citation_counts.txt".format(RESULTS_FOLDER)
    word_citation_counts_file = "{}/word_citation_counts.txt".format(RESULTS_FOLDER)
    assign_c_file = "{}/assign_c.txt".format(RESULTS_FOLDER)

    if not os.path.exists(citation_counts_file):
        citation_counts = read_and_process_c(assign_c_file,
                                             lambda x, y, z: cited_this_iteration(x, y, len(idx2src_sub)),
                                             (len(idx2src_sub),),
                                             edges,
                                             subreddits
                          )
        write2D(citation_counts_file, citation_counts)
    else:
        citation_counts = read2D(citation_counts_file)
    #TO-DO: Pull top-k cites per doc, and top-k words per cite per doc
    cites_per_doc = aggregate_and_top_k(citation_counts,
                                        20,
                                        (len(idx2src_sub), ),
                                        lambda x, y: top_k_cite(x, y, idx2src_sub),
                                        subreddits,
                                        idx2tgt_sub
                    )
    with open("{}/top_k_cites.json".format(OUTPUTS_FOLDER), "w+") as f:
        f.write(json.dumps(cites_per_doc, indent=4))
    
    #to save space we will do subsequent analysis with ONLY the src subreddits that appear in the current top-k most cited
    old_src_idx2new, new2old_src_idx = filter_and_remap(cites_per_doc, src_sub2idx)
    if not os.path.exists(word_citation_counts_file):
        word_citation_counts = read_and_process_c(assign_c_file,
                                                 lambda x, y, z: words_by_source(x, y, len(old_src_idx2new), len(idx2vocab), old_src_idx2new, z),
                                                  (len(new2old_src_idx), len(idx2vocab)),
                                                  edges,
                                                  subreddits,
                                                  tgt_blobs
                               )
        write3D(word_citation_counts_file, word_citation_counts)
    else:
        word_citation_counts = read3D(word_citation_counts_file)
        print("READ WORD-CITATION")

    words_per_cite_per_doc = aggregate_and_top_k(word_citation_counts,
                                                  20,
                                                  (len(idx2src_sub), len(idx2vocab)),
                                                  lambda x, y: top_k_word_cite(x, y, idx2src_sub, idx2vocab, tgt_blobs, new2old_src_idx),
                                                  subreddits,
                                                  idx2tgt_sub
                             )
    with open("{}/top_k_words_per_cite.json".format(OUTPUTS_FOLDER), "w+") as f:
        f.write(json.dumps(words_per_cite_per_doc, indent=4))
