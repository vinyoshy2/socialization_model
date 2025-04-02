#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include "model.h"
#include "utils.h"
#include "dataset.h"


// Constructor
CollapsedGibbsSocLDA::CollapsedGibbsSocLDA(const TextNetwork& text_network, int n_topic) 
    : text_network(text_network), V(text_network.vocab_size), k(n_topic),
        src_M(text_network.src_blobs.size()), tgt_M(text_network.tgt_blobs.size()),
        src_L(text_network.num_src_subreddits), tgt_L(text_network.num_tgt_subreddits),
        alpha_phi(1.0), alpha_theta(1.0), alpha_gamma(1.0), alpha_psi(1.0),
        lambda_theta(1.0), lambda_psi(1.0) {
    
    // prepare rng
    std::mt19937 gen(rd());

    // Initialize src_N and tgt_N
    src_N.resize(tgt_M);
    tgt_N.resize(tgt_M);
    for (int i = 0; i < tgt_M; ++i) {
        src_N[i] = text_network.tgt_blobs[i].size();
        tgt_N[i] = text_network.tgt_blobs[i].size();
    } 

    // Initialize count matrices
    dc.resize(tgt_M, std::vector<int>(src_L + 1, 0));
    ct.resize(src_L + 1, std::vector<int>(k, 0));
    rts.resize(tgt_L, std::vector<std::vector<int>>(k, std::vector<int>(2, 0)));
    c_t_.resize(src_M, std::vector<int>(k, 0));
    wt.resize(V, std::vector<int>(k, 0));
    forced_innovation_count.resize(tgt_L, 0);

    // Initialize matrix row/column sum counts
    c_sum.resize(src_L + 1, 0);
    d_cited_sum.resize(tgt_M, 0);
    r0_sum.resize(tgt_L, 0);
    r1_sum.resize(tgt_L, 0);
    t_sum.resize(k, 0);
}

// Gibbs sampling function
void CollapsedGibbsSocLDA::run_gibbs(int n_gibbs, bool verbose) {
    // Initialize Gibbs sampler
    init_gibbs(n_gibbs);

    if (verbose) {
        std::cout << "\n========== START SAMPLER ==========" << std::endl;
    }

    // Run Gibbs sampler
    for (int iter = 0; iter < n_gibbs; ++iter) {

        // Update source subreddit documents
        for (int c_ = 0; c_ < src_M; ++c_) {
            for (int n = 0; n < src_N[c_]; ++n) {
                update_t_(c_, n, iter);
            }
        }

        // Update target subreddit documents
        for (int d = 0; d < tgt_M; ++d) {
            for (int n = 0; n < tgt_N[d]; ++n) {
                int r = text_network.tgt_subreddits[d];
                update_t(d, n, r, iter, iter);
                update_cs(d, n, r, iter, iter + 1);
            }
        }

        // Print progress every 200 iterations
        if (verbose && (iter + 1) % 200 == 0) {
            std::cout << "\n===== ITERATION " << iter << " =====" << std::endl;
        }
    }
}

std::vector<std::vector<std::vector<double>>> CollapsedGibbsSocLDA::recover_gamma(int total_iter, int num_warmup) {
    
    std::vector<std::vector<std::vector<double>>> gamma(
        tgt_M, std::vector<std::vector<double>>(src_L, std::vector<double>(total_iter - num_warmup, 0.0)));
    std::vector<std::vector<std::vector<double>>> tmp_counts(
        tgt_M, std::vector<std::vector<double>>(src_L, std::vector<double>(total_iter - num_warmup, 0.0)));

    // Collect counts from samples
    for (int d = 0; d < tgt_M; ++d) {
        for (int n = 0; n < tgt_N[d]; ++n) {
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                if (assign_s[d][n][iter] == 0) {
                    tmp_counts[d][assign_c[d][n][iter]][iter - num_warmup] += 1.0;
                }
            }
        }
    }
    // Compute gamma
    for (int d = 0; d < tgt_M; ++d) {
        int num_edges = text_network.edges[d].size();
        for (size_t i = 0; i < num_edges; ++i) {
            int edge = text_network.edges[d][i];
            double sum_val = 0.0;
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                double numerator = tmp_counts[d][edge][iter - num_warmup] + alpha_gamma;
                /*double denominator = 0.0;
                for (int j = 0; j < src_L; ++j) {
                    denominator += tmp_counts[d][j][iter - num_warmup];
                }
                denominator += num_edges * alpha_gamma;*/
                gamma[d][edge][iter-num_warmup] = numerator;
                /// denominator;
            }
        }
    }
    return gamma;
}

std::vector<std::vector<std::vector<double>>> CollapsedGibbsSocLDA::recover_psi(int total_iter, int num_warmup) {

    std::vector<std::vector<std::vector<double>>> psi(
        tgt_L, std::vector<std::vector<double>>(k, std::vector<double>(total_iter - num_warmup, 0.0)));
    std::vector<std::vector<std::vector<double>>> tmp_counts(
        tgt_L, std::vector<std::vector<double>>(k, std::vector<double>(total_iter - num_warmup, 0.0)));

    // Collect counts from samples
    for (int d = 0; d < tgt_M; ++d) {
        int r = text_network.tgt_subreddits[d];
        for (int n = 0; n < tgt_N[d]; ++n) {
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                if (assign_s[d][n][iter] == 1) {
                    tmp_counts[r][assign_t[d][n][iter]][iter - num_warmup] += 1.0;
                }
            }
        }
    }

    // Compute psi
    for (int d = 0; d < tgt_M; ++d) {
        int r = text_network.tgt_subreddits[d];
        for (int topic = 0; topic < k; ++topic) {
            double sum_val = 0.0;
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                double numerator = tmp_counts[r][topic][iter - num_warmup] + alpha_psi;
                /*double denominator = 0.0;
                for (int j = 0; j < k; ++j) {
                    denominator += tmp_counts[d][j][iter - num_warmup];
                }
                denominator += k * alpha_psi;*/
                psi[r][topic][iter - num_warmup]= numerator;
                // / denominator;
            }
        }
    }
    return psi;
}

std::vector<std::vector<std::vector<double>>> CollapsedGibbsSocLDA::recover_phi(int total_iter, int num_warmup) {

    std::vector<std::vector<std::vector<double>>> phi(
        k, std::vector<std::vector<double>>(V, std::vector<double>(total_iter - num_warmup, 0.0)));
    std::vector<std::vector<std::vector<double>>> tmp_counts(
        k, std::vector<std::vector<double>>(V, std::vector<double>(total_iter - num_warmup, 0.0)));

    // Collect counts from target network
    for (int d = 0; d < tgt_M; ++d) {
        for (int n = 0; n < tgt_N[d]; ++n) {
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                int cur_topic = assign_t[d][n][iter];
                int cur_word = text_network.tgt_blobs[d][n];
                tmp_counts[cur_topic][cur_word][iter - num_warmup] += 1.0;
            }
        }
    }

    // Collect counts from source network
    for (int d = 0; d < src_M; ++d) {
        for (int n = 0; n < src_N[d]; ++n) {
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                int cur_topic = assign_t_[d][n][iter];
                int cur_word = text_network.src_blobs[d][n]; 
                tmp_counts[cur_topic][cur_word][iter - num_warmup] += 1.0;
            }
        }
    }

    // Compute phi
    for (int t = 0; t < k; ++t) {
        for (int w = 0; w < V; ++w) {
            double sum_val = 0.0;
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                double numerator = tmp_counts[t][w][iter - num_warmup] + alpha_phi;
                /*double denominator = 0.0;
                for (int j = 0; j < V; ++j) {
                    denominator += tmp_counts[t][j][iter - num_warmup];
                }
                denominator += V * alpha_phi;*/
                phi[t][w][iter-num_warmup] = numerator;
                // / denominator;
            }
        }
    }
    return phi;
}

std::vector<std::vector<std::vector<double>>> CollapsedGibbsSocLDA::recover_theta(int total_iter, int num_warmup) {
    std::vector<std::vector<std::vector<double>>> theta(
        src_M, std::vector<std::vector<double>>(k, std::vector<double>(total_iter - num_warmup, 0.0)));
    std::vector<std::vector<std::vector<double>>> tmp_counts(
        src_M, std::vector<std::vector<double>>(k, std::vector<double>(total_iter - num_warmup, 0.0)));

    // Collect counts from source network
    for (int d = 0; d < src_M; ++d) {
        for (int n = 0; n < src_N[d]; ++n) {
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                int cur_topic = assign_t_[d][n][iter];
                tmp_counts[d][cur_topic][iter - num_warmup] += 1.0;
            }
        }
    }

    // Collect counts from target network, considering connections
    for (int d = 0; d < tgt_M; ++d) {
        for (int n = 0; n < tgt_N[d]; ++n) {
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                int cur_topic = assign_t[d][n][iter];
                int cur_c = assign_c[d][n][iter];
                if (cur_c != src_L) {  // Ensure valid source index
                    tmp_counts[cur_c][cur_topic][iter - num_warmup] += 1.0;
                }
            }
        }
    }

    // Compute theta
    for (int d = 0; d < src_M; ++d) {
        for (int t = 0; t < k; ++t) {
            double sum_val = 0.0;
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                double numerator = tmp_counts[d][t][iter - num_warmup] + alpha_theta;
                /*double denominator = 0.0;
                for (int j = 0; j < k; ++j) {
                    denominator += tmp_counts[d][j][iter - num_warmup];
                }
                denominator += alpha_theta * k;*/
                theta[d][t][iter-num_warmup] = numerator;
                // / denominator;
            }
        }
    }
    return theta;
}

std::vector<std::vector<std::vector<double>>> CollapsedGibbsSocLDA::recover_lambda(int total_iter, int num_warmup) {
    std::vector<std::vector<std::vector<double>>> lambdas(
        tgt_L, std::vector<std::vector<double>>(2, std::vector<double>(total_iter - num_warmup, 0.0)));
    std::vector<std::vector<std::vector<double>>> tmp_counts(
        tgt_L, std::vector<std::vector<double>>(2, std::vector<double>(total_iter - num_warmup, 0.0)));

    // Collect counts from target network
    for (int d = 0; d < tgt_M; ++d) {
        int subreddit = text_network.tgt_subreddits[d];  // Get subreddit index
        for (int n = 0; n < tgt_N[d]; ++n) {
            for (int iter = num_warmup; iter < total_iter; ++iter) {
                int cur_s = assign_s[d][n][iter];
                tmp_counts[subreddit][cur_s][iter - num_warmup] += 1.0;
            }
        }
    }

    // Compute lambda values
    for (int r = 0; r < tgt_L; ++r) {
        double sum_cite = 0;
        double sum_inno = 0;
        for (int iter = num_warmup; iter < total_iter; ++iter) {
            /*double denom = 0.0;
            for (int j = 0; j < 2; ++j) {
                denom += tmp_counts[r][j][iter - num_warmup];
            }
            denom -= forced_innovation_count[r];
            denom += lambda_theta + lambda_psi;*/
            lambdas[r][0][iter - num_warmup] += (tmp_counts[r][0][iter - num_warmup] + lambda_theta);
            // / denom;
            lambdas[r][1][iter - num_warmup] += (tmp_counts[r][1][iter - num_warmup] - forced_innovation_count[r] + lambda_psi);
            // / denom;
        }
    }
    return lambdas;
}




// Initialize the Gibbs sampler
void CollapsedGibbsSocLDA::init_gibbs(int n_gibbs) {
    
    // Max lengths
    int src_N_max = *max_element(src_N.begin(), src_N.end());
    int tgt_N_max = *max_element(tgt_N.begin(), tgt_N.end());

    // Resize assignment matrices
    assign_c.resize(tgt_M, std::vector<std::vector<int>>(tgt_N_max, std::vector<int>(n_gibbs + 1, 0)));
    assign_s.resize(tgt_M, std::vector<std::vector<int>>(tgt_N_max, std::vector<int>(n_gibbs + 1, 0)));
    assign_t.resize(tgt_M, std::vector<std::vector<int>>(tgt_N_max, std::vector<int>(n_gibbs + 1, 0)));
    assign_t_.resize(src_M, std::vector<std::vector<int>>(src_N_max, std::vector<int>(n_gibbs + 1, 0)));

    // Reset count matrices
    for (auto& row : c_t_) fill(row.begin(), row.end(), 0);
    for (auto& row : dc) fill(row.begin(), row.end(), 0);
    for (auto& row : ct) fill(row.begin(), row.end(), 0);
    for (auto& row : wt) fill(row.begin(), row.end(), 0);
    for (auto& matrix : rts)
        for (auto& row : matrix)
            fill(row.begin(), row.end(), 0);

    fill(forced_innovation_count.begin(), forced_innovation_count.end(), 0);
    fill(c_sum.begin(), c_sum.end(), 0);
    fill(d_cited_sum.begin(), d_cited_sum.end(), 0);
    fill(r0_sum.begin(), r0_sum.end(), 0);
    fill(r1_sum.begin(), r1_sum.end(), 0);
    fill(t_sum.begin(), t_sum.end(), 0);

    // Random number generator
    std::uniform_int_distribution<int> topic_dist(0, k - 1);
    std::uniform_int_distribution<int> binary_dist(0, 1);

    // Initialize values for each src comment
    for (int d = 0; d < src_M; ++d) {
        for (int n = 0; n < src_N[d]; ++n) {
            int w_dn = text_network.src_blobs[d][n];
            int cur_topic = topic_dist(gen);
            assign_t_[d][n][0] = cur_topic;

            // Increment counters
            wt[w_dn][cur_topic]++;
            c_t_[d][cur_topic]++;
            t_sum[cur_topic]++;
        }
    }

    // Initialize values for each tgt comment
    for (int d = 0; d < tgt_M; ++d) {
        int r = text_network.tgt_subreddits[d];
        for (int n = 0; n < tgt_N[d]; ++n) {
            int w_dn = text_network.tgt_blobs[d][n];

            // Assign innovation flag (s)
            if (text_network.edges[d].empty()) {
                assign_s[d][n][0] = 1;
                forced_innovation_count[r]++;
            } else {
                assign_s[d][n][0] = binary_dist(gen);
            }

            // Assign source subreddit (c)
            if (assign_s[d][n][0] == 0) {
                std::uniform_int_distribution<int> edge_dist(0, text_network.edges[d].size() - 1);
                assign_c[d][n][0] = text_network.edges[d][edge_dist(gen)];
            } else {
                assign_c[d][n][0] = src_L;
            }

            // Assign topic (t)
            assign_t[d][n][0] = topic_dist(gen);

            // Increment counters
            int cur_t = assign_t[d][n][0];
            int cur_s = assign_s[d][n][0];
            int cur_c = assign_c[d][n][0];

            dc[d][cur_c]++;
            ct[cur_c][cur_t]++;
            rts[r][cur_t][cur_s]++;
            wt[w_dn][cur_t]++;
            c_sum[cur_c]++;
            t_sum[cur_t]++;
            if (cur_s == 0) {
                d_cited_sum[d]++;
                r0_sum[r]++;
            } else {
                r1_sum[r]++;
            }
        }
    }
}

std::vector<double> CollapsedGibbsSocLDA::conditional_prob_cs(int w_dn, int d, int r, int t) {
    size_t edge_count = text_network.edges[d].size();
    std::vector<double> prob(edge_count + 1, 0.0);

    for (size_t ind = 0; ind < edge_count; ind++) {
        int i = text_network.edges[d][ind];

        double _1 = (c_t_[i][t] + ct[i][t] + alpha_theta) / (src_N[i] + c_sum[i] + k * alpha_theta);
        double _2 = (dc[d][i] + alpha_gamma) / (d_cited_sum[d] + edge_count * alpha_gamma);
        double _3 = (r0_sum[r] + lambda_theta)
                    / (r0_sum[r] + r1_sum[r] - forced_innovation_count[r] + lambda_theta + lambda_psi);

        prob[ind] = _1 * _2 * _3;
    }

    double _1 = (rts[r][t][1] + alpha_psi) / (r1_sum[r] + k * alpha_psi);
    double _2 = (r1_sum[r] - forced_innovation_count[r] + lambda_psi) 
                / (r0_sum[r] + r1_sum[r] - forced_innovation_count[r] + lambda_theta + lambda_psi);
    prob[edge_count] = _1 * _2;

    double prob_sum = std::accumulate(prob.begin(), prob.end(), 0.0);
    for (double& p : prob) p /= prob_sum;

    return prob;
}

std::vector<double> CollapsedGibbsSocLDA::conditional_prob_t(int w_dn, int d, int r, int c, int s) {
    std::vector<double> prob(k, 0.0);

    for (int i = 0; i < k; i++) {
        double _1 = (wt[w_dn][i] + alpha_phi) / (t_sum[i] + V * alpha_phi);
        double _2;
        if (s == 0) {
            _2 = (c_t_[c][i] + ct[c][i] + alpha_theta) / (src_N[c] + c_sum[c] + k * alpha_theta);
        } else {
            _2 = (rts[r][i][1] + alpha_psi) / (r1_sum[r] + k * alpha_psi);
        }
        prob[i] = _1 * _2;
    }

    double prob_sum = std::accumulate(prob.begin(), prob.end(), 0.0);
    for (double& p : prob) p /= prob_sum;

    return prob;
}

std::vector<double> CollapsedGibbsSocLDA::conditional_prob_t_(int w_c_n, int c_) {
    std::vector<double> prob(k, 0.0);

    for (int i = 0; i < k; i++) {
        double _1 = (wt[w_c_n][i] + alpha_phi) /
                    (t_sum[i] + V * alpha_phi);
        double _2 = (c_t_[c_][i] + ct[c_][i] + alpha_theta) /
                    (src_N[c_] + c_sum[c_] + k * alpha_theta);
        prob[i] = _1 * _2;
    }

    double prob_sum = std::accumulate(prob.begin(), prob.end(), 0.0);
    for (double& p : prob) p /= prob_sum;

    return prob;
}


void CollapsedGibbsSocLDA::update_cs(int d, int n, int r, int cs_iter, int t_iter) {
    if (text_network.edges[d].empty()) {
        assign_c[d][n][cs_iter + 1] = assign_c[d][n][cs_iter];
        assign_s[d][n][cs_iter + 1] = assign_s[d][n][cs_iter];
        return;
    }

    int w_dn = text_network.tgt_blobs[d][n];
    const std::vector<int>& edges = text_network.edges[d];

    int i_t = assign_t[d][n][t_iter];
    int i_c = assign_c[d][n][cs_iter];
    int i_s = assign_s[d][n][cs_iter];

    // Decrement counters
    dc[d][i_c]--;
    rts[r][i_t][i_s]--;
    ct[i_c][i_t]--;
    c_sum[i_c]--;
    if (i_s == 0) {
        d_cited_sum[d]--;
        r0_sum[r]--;
    } else {
        r1_sum[r]--;
    }

    // Compute new assignment probabilities
    std::vector<double> prob = conditional_prob_cs(w_dn, d, r, i_t);
    int result = weighted_sample(prob, gen);

    int new_s = (result == edges.size()) ? 1 : 0;
    int new_c = (new_s == 1) ? src_L : edges[result];

    // Increment counters
    dc[d][new_c]++;
    rts[r][i_t][new_s]++;
    ct[new_c][i_t]++;
    c_sum[new_c]++;
    if (new_s == 0) {
        d_cited_sum[d]++;
        r0_sum[r]++;
    } else {
        r1_sum[r]++;
    }

    assign_c[d][n][cs_iter + 1] = new_c;
    assign_s[d][n][cs_iter + 1] = new_s;
}

void CollapsedGibbsSocLDA::update_t(int d, int n, int r, int cs_iter, int t_iter) {
    int w_dn = text_network.tgt_blobs[d][n];

    int i_t = assign_t[d][n][t_iter];
    int i_c = assign_c[d][n][cs_iter];
    int i_s = assign_s[d][n][cs_iter];

    // Decrement counters
    rts[r][i_t][i_s]--;
    ct[i_c][i_t]--;
    wt[w_dn][i_t]--;
    t_sum[i_t]--;

    // Compute new assignment probabilities
    std::vector<double> prob = conditional_prob_t(w_dn, d, r, i_c, i_s);
    int i_tp1 = weighted_sample(prob, gen);

    // Increment counters
    rts[r][i_tp1][i_s]++;
    ct[i_c][i_tp1]++;
    wt[w_dn][i_tp1]++;
    t_sum[i_tp1]++;

    assign_t[d][n][t_iter + 1] = i_tp1;
}

void CollapsedGibbsSocLDA::update_t_(int c_, int n, int t_iter) {
    int w_dn = text_network.src_blobs[c_][n];
    int i_t_ = assign_t_[c_][n][t_iter];

    // Decrement counters
    c_t_[c_][i_t_]--;
    wt[w_dn][i_t_]--;
    t_sum[i_t_]--;

    // Compute new assignment probabilities
    std::vector<double> prob = conditional_prob_t_(w_dn, c_);
    int i_tp1 = weighted_sample(prob, gen);

    // Increment counters
    c_t_[c_][i_tp1]++;
    wt[w_dn][i_tp1]++;
    t_sum[i_tp1]++;

    assign_t_[c_][n][t_iter + 1] = i_tp1;
}