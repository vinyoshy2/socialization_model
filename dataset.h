#ifndef DATASET
#define DATASET

#include <vector>

struct TextNetwork {
    std::vector<std::vector<int>> src_blobs;
    std::vector<std::vector<int>> tgt_blobs;
    std::vector<std::vector<int>> edges;
    std::vector<int> tgt_subreddits;
    std::vector<int> tgt_subgroups;
    int vocab_size;
    int num_src_subreddits;
    int num_tgt_subreddits;
    int num_tgt_subgroups;
};

#endif
