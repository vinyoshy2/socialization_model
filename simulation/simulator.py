import numpy as np
import json
import sys
import random 

class TextBlob:
    def _preprocess(self, text):
        return text.lower().split()
    def __init__(self, text):
        self.text = self._preprocess(text)

class TextNetwork:
    def compute_vocab_size(self, src_blobs, tgt_blobs):
        num_words = 0
        for blob in src_blobs:
            for word in blob:
                num_words = max(num_words, word)
        for blob in tgt_blobs:
            for word in blob:
                num_words = max(num_words, word)
        return num_words+1
    def __init__(self, src_blobs, tgt_blobs, edges, subreddits, subgroups):
        """
            src_blobs: List of lists. Each list contains words associated with a particular source subreddit
            tgt_blobs: List of lists. Each list contains words associated with a particular target subreddit,user pair
            edges: List of lists. List i contains the list of src_blob indices that target blob i is connected to
            #subreddits: list of the subreddit associated with each tgt_blob in tgt_blobs
        """
        self.src_blobs = src_blobs
        self.tgt_blobs = tgt_blobs
        self.edges = edges
        self.subreddits = subreddits
        self.subgroups = subgroups
        self.vocab_size = self.compute_vocab_size(src_blobs, tgt_blobs)
        self.num_src_subreddits = len(src_blobs)
        self.num_tgt_subreddits = max(subreddits) + 1
        self.num_tgt_groups = max(subgroups) + 1

    def write_2d_mat(self, loc, mat):
        with open(loc, "w+") as f:
            f.write("{}\n".format(len(mat)))
            for i in range(len(mat)):
                f.write("{}\n".format(len(mat[i])))
            for i in range(len(mat)):
                f.write("{}\n".format(" ".join([str(j) for j in mat[i]])))
        return
    def write_1d_mat(self, loc, mat):
        with open(loc, "w+") as f:
            f.write("{}\n".format(len(mat)))
            f.write("{}\n".format(" ".join([str(j) for j in mat])))
        return

    def write_to_files(self, folder):
        self.write_2d_mat(folder + "/src_blobs.txt", self.src_blobs)
        self.write_2d_mat(folder + "/tgt_blobs.txt", self.tgt_blobs)
        self.write_2d_mat(folder + "/edges.txt", self.edges)
        self.write_1d_mat(folder + "/subreddits.txt", self.subreddits)
        self.write_1d_mat(folder + "/subgroups.txt", self.subgroups)


class ModelSpecification:
    def __init__(self, num_src_docs, src_doc_sizes, num_tgt_groups, num_tgt_subreddits, num_tgt_comments, tgt_comment_sizes,
                 num_topics, vocab_size, vocab_vectors, src_topic_vectors, tgt_subreddit_topic_vectors,
                 edge_list, edge_weights, coin_flip_probs):
        self.num_src_docs = num_src_docs
        self.src_doc_sizes = src_doc_sizes
        self.num_tgt_subreddits = num_tgt_subreddits
        self.num_tgt_groups_per_sub = num_tgt_groups
        self.num_tgt_comments_per_group = num_tgt_comments
        self.tgt_comment_sizes = tgt_comment_sizes
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.vocab_vectors = vocab_vectors
        self.src_topic_vectors = src_topic_vectors
        self.tgt_subreddit_topic_vectors = tgt_subreddit_topic_vectors
        self.edge_list = edge_list
        self.edge_weights = edge_weights
        self.coin_flip_probs = coin_flip_probs
    def to_file(self, loc):
        model_params = {
            "num_src_docs": self.num_src_docs,
            "src_doc_sizes": self.src_doc_sizes,
            "num_tgt_subreddits": self.num_tgt_subreddits,
            "num_tgt_groups_per_sub": self.num_tgt_groups_per_sub,
            "num_tgt_comments_per_group": self.num_tgt_comments_per_group,
            "tgt_comment_sizes": self.tgt_comment_sizes,
            "self.num_topics": self.num_topics,
            "vocab_size": self.vocab_size,
            "vocab_vectors":  self.vocab_vectors,
            "src_topic_vectors": self.src_topic_vectors,
            "tgt_subreddit_topic_vectors": self.tgt_subreddit_topic_vectors,
            "edge_list":  self.edge_list,
            "edge_weights":  self.edge_weights,
            "coin_flip_probs":  self.coin_flip_probs
        }
        with open(loc, "w+") as f:
            f.write(json.dumps(model_params))
        return model_params

        
def genNetwork(model_spec):
    src_blobs = []
    # generate source documents
    print("Generating source docs...")
    for i in range(0, model_spec.num_src_docs):
        if i % 20 == 0:
            print("{} out of {}...".format(i, model_spec.num_src_docs))
        cur_blob_size = model_spec.src_doc_sizes[i]
        cur_blob = []
        for j in range(0, cur_blob_size):
            cur_topic = np.random.choice(list(range(model_spec.num_topics)),  p = model_spec.src_topic_vectors[i])
            cur_word = np.random.choice(list(range(model_spec.vocab_size)), p = model_spec.vocab_vectors[cur_topic])
            cur_blob.append(cur_word)
        src_blobs.append(cur_blob)
    # generate tgt comments
    tgt_blobs = []
    tgt_subreddits = []
    tgt_groups = [] 
    group_ind = 0
    doc_ind = 0 
    total_docs = 0

    #compute total number of docs
    for i in range(0, model_spec.num_tgt_subreddits):
        for j in range(0, model_spec.num_tgt_groups_per_sub[i]):
            for k in range(0, model_spec.num_tgt_comments_per_group[j]):
                total_docs += 1
    
    print("Generating target docs...")
    for i in range(0, model_spec.num_tgt_subreddits):
        for j in range(0, model_spec.num_tgt_groups_per_sub[i]):
            for k in range(0, model_spec.num_tgt_comments_per_group[j]):
                if doc_ind % 100 == 0:
                    print("{} out of {}...".format(doc_ind, total_docs))
                tgt_subreddits.append(i)
                tgt_groups.append(group_ind)
                cur_blob_size = model_spec.tgt_comment_sizes[doc_ind]
                cur_blob = []
                for k in range(0, cur_blob_size):
                    innovate = np.random.binomial(1, model_spec.coin_flip_probs[group_ind])
                    if innovate == 1 or len(model_spec.edge_list[doc_ind]) == 0:
                        topic_vector = model_spec.tgt_subreddit_topic_vectors[i]
                    else:
                        src_subreddit = np.random.choice(model_spec.edge_list[doc_ind], p=model_spec.edge_weights[doc_ind])
                        topic_vector = model_spec.src_topic_vectors[src_subreddit]
                    cur_topic = np.random.choice(list(range(model_spec.num_topics)), p = topic_vector)
                    vocab_vector = model_spec.vocab_vectors[cur_topic]
                    cur_word = np.random.choice(list(range(model_spec.vocab_size)), p = vocab_vector)
                    cur_blob.append(cur_word)
                doc_ind += 1
                tgt_blobs.append(cur_blob)
            group_ind +=1
    return TextNetwork(src_blobs, tgt_blobs, model_spec.edge_list, tgt_subreddits, tgt_groups)

def gen_specification(src_subs, tgt_subs, tgt_groups_per_sub, authors_per_tgt_group, vocab_size, src_doc_size,
                      tgt_doc_size, num_topics, edge_max):
 
    src_doc_sizes = [src_doc_size for i in range(0, src_subs)]
    tgt_comment_sizes = [tgt_doc_size for i in range(0,
        tgt_subs*tgt_groups_per_sub*authors_per_tgt_group)
    ]
    num_tgt_groups = [tgt_groups_per_sub for i in range(0, tgt_subs)]
    num_tgt_comments = [authors_per_tgt_group for i in range(0, tgt_subs*tgt_groups_per_sub)]
 
    phis = np.random.dirichlet([1/3 for i in range(0, vocab_size)], num_topics).tolist()
    thetas = np.random.dirichlet([1/3 for i in range(0, num_topics)], src_subs).tolist()
    psis = np.random.dirichlet([1/3 for i in range(0, num_topics)], tgt_subs).tolist()
    lambdas = np.random.beta(1,1,tgt_subs*tgt_groups_per_sub).tolist()
    edge_lists = []
    gammas = []
    for tgt_sub in range(tgt_subs):
        for group in range(tgt_groups_per_sub):
            for author in range(authors_per_tgt_group):
                num_edges = np.random.randint(0, edge_max)
                cur_edges = random.sample(list(range(src_subs)), k=num_edges)
                cur_edges.sort()
                edge_lists.append(cur_edges)
                gammas.append(np.random.dirichlet([1/3 for i in range(len(cur_edges))]).tolist())
    return ModelSpecification(
                num_src_docs=src_subs,
                src_doc_sizes=src_doc_sizes,
                num_tgt_subreddits=tgt_subs,
                num_tgt_groups=num_tgt_groups,
                num_tgt_comments=num_tgt_comments,
                tgt_comment_sizes=tgt_comment_sizes,
                num_topics=num_topics,
                vocab_size=vocab_size,
                vocab_vectors=phis,
                src_topic_vectors=thetas,
                tgt_subreddit_topic_vectors=psis,
                edge_list=edge_lists,
                edge_weights=gammas,
                coin_flip_probs=lambdas
            )

def main():
    dir_name = sys.argv[1]
    model_spec = gen_specification(src_subs=10, tgt_subs=5, tgt_groups_per_sub = 2, authors_per_tgt_group=50, vocab_size=500, src_doc_size=500, tgt_doc_size=100, num_topics=10, edge_max=10)

    textNetwork = genNetwork(model_spec)
    textNetwork.write_to_files("{}".format(dir_name))
    model_spec.to_file("{}/params.json".format(dir_name))
    return 
if __name__ == "__main__": 
    main()

