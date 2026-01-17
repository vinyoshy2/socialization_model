#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include "model.h"
#include "utils.h"
#include "dataset.h"


// Constructor
CollapsedGibbsSocLDA::CollapsedGibbsSocLDA(const TextNetwork& text_network, int n_topic, float alpha_sum_topics, float alpha_sum_vocab, float alpha_sum_edges, const std::string& out_dir) 
    : text_network(text_network), V(text_network.vocab_size), k(n_topic),
        src_M(text_network.src_blobs.size()), tgt_M(text_network.tgt_blobs.size()),
        src_L(text_network.num_src_subreddits), tgt_L(text_network.num_tgt_subreddits),
        alpha_phi(alpha_sum_vocab / text_network.vocab_size), alpha_theta(alpha_sum_topics / n_topic), alpha_psi(alpha_sum_topics / n_topic), alpha_sum_edges(alpha_sum_edges),
        lambda_theta(1.0), lambda_psi(1.0), output_dir(out_dir) {
    
    // prepare rng
    gen = std::mt19937(std::random_device{}());
   

    // Initialize src_N and tgt_N
    src_N.resize(src_M);
    tgt_N.resize(tgt_M);
    for (int i = 0; i < src_M; ++i) {
        src_N[i] = text_network.src_blobs[i].size();
    }
    for (int i = 0; i < tgt_M; ++i) {
        tgt_N[i] = text_network.tgt_blobs[i].size();
    } 

    std::cout << "n_topic: " << n_topic << std::endl;
    std::cout << "tgt_M: " << tgt_M << std::endl;
    std::cout << "src_L: " << src_L << std::endl;
    std::cout << "src_M: " << src_M << std::endl;
    std::cout << "tgt_L: " << tgt_L << std::endl;
    std::cout << "V: " << V << std::endl;
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
void CollapsedGibbsSocLDA::run_gibbs(int n_gibbs, int n_warmup, bool verbose) {
    // Initialize Gibbs sampler
    init_gibbs(n_gibbs);

    if (verbose) {
        std::cout << "\n========== START SAMPLER ==========" << std::endl;
    }

    // Set up timer
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    // Run Gibbs sampler
    for (int iter = 0; iter < n_gibbs; ++iter) {
        
        // Update source subreddit documents
        for (int c_ = 0; c_ < src_M; ++c_) {
            for (int n = 0; n < src_N[c_]; ++n) {
                update_t_(c_, n);
            }
        }

        // Update target subreddit documents
        for (int d = 0; d < tgt_M; ++d) {
            for (int n = 0; n < tgt_N[d]; ++n) {
                int r = text_network.tgt_subreddits[d];
                update_t(d, n, r);
                update_cs(d, n, r);
            }
        }
        
        // Print progress every 200 iterations
        if (verbose && (iter + 1) % 200 == 0) {
            std::cout << "\n===== ITERATION " << iter << " =====" << std::endl;
            auto t2 = high_resolution_clock::now();
            duration<double, std::milli> ms_double = t2 - t1;
            std::cout << ms_double.count() << "ms\n";
            t1 = high_resolution_clock::now();
        }

        if (iter >= n_warmup) {
            //Recover parameters
            std::vector<std::vector<double>> gamma = recover_gamma();
            std::vector<std::vector<double>> psi = recover_psi();
            std::vector<std::vector<double>> phi = recover_phi();
            std::vector<std::vector<double>> theta = recover_theta();
            std::vector<std::vector<double>> lambda = recover_lambda();

            //Save to parameters to output file   
            append2D(output_dir + "/gamma.txt", gamma);
            append2D(output_dir + "/psi.txt", psi);
            append2D(output_dir + "/phi.txt", phi);
            append2D(output_dir + "/theta.txt", theta);
            append2D(output_dir + "/lambda.txt", lambda);
        }
    }
}

std::vector<std::vector<double>> CollapsedGibbsSocLDA::recover_gamma() {
    
    std::vector<std::vector<double>> gamma(tgt_M, std::vector<double>(src_L, 0.0));
    std::vector<std::vector<double>> tmp_counts(tgt_M, std::vector<double>(src_L, 0.0));

    // Collect counts from samples
    for (int d = 0; d < tgt_M; ++d) {
        for (int n = 0; n < tgt_N[d]; ++n) {
            if (assign_s[d][n] == 0) {
                tmp_counts[d][assign_c[d][n]] += 1.0;
            }
        }
    }
    // Compute gamma
    for (int d = 0; d < tgt_M; ++d) {
        int num_edges = text_network.edges[d].size();
        for (size_t i = 0; i < num_edges; ++i) {
            int edge = text_network.edges[d][i];
            // double sum_val = 0.0;
            double numerator = tmp_counts[d][edge] + (alpha_sum_edges/num_edges);
            /*double denominator = 0.0;
            for (int j = 0; j < src_L; ++j) {
                denominator += tmp_counts[d][j][iter - num_warmup];
            }
            denominator += num_edges * alpha_gamma;*/
            gamma[d][edge] = numerator;
            /// denominator;
        }
    }
    return gamma;
}

std::vector<std::vector<double>> CollapsedGibbsSocLDA::recover_psi() {

    std::vector<std::vector<double>> psi(tgt_L, std::vector<double>(k, 0.0));
    std::vector<std::vector<double>> tmp_counts(tgt_L, std::vector<double>(k, 0.0));

    // Collect counts from samples
    for (int d = 0; d < tgt_M; ++d) {
        int r = text_network.tgt_subreddits[d];
        for (int n = 0; n < tgt_N[d]; ++n) {
            if (assign_s[d][n] == 1) {
                tmp_counts[r][assign_t[d][n]] += 1.0;
            }
        }
    }

    // Compute psi
    for (int d = 0; d < tgt_M; ++d) {
        int r = text_network.tgt_subreddits[d];
        for (int topic = 0; topic < k; ++topic) {
            // double sum_val = 0.0;
            double numerator = tmp_counts[r][topic] + alpha_psi;
            /*double denominator = 0.0;
            for (int j = 0; j < k; ++j) {
                denominator += tmp_counts[d][j][iter - num_warmup];
            }
            denominator += k * alpha_psi;*/
            psi[r][topic] = numerator;
            // / denominator;
        }
    }
    return psi;
}

std::vector<std::vector<double>> CollapsedGibbsSocLDA::recover_phi() {

    std::vector<std::vector<double>> phi(k, std::vector<double>(V, 0.0));
    std::vector<std::vector<double>> tmp_counts(k, std::vector<double>(V, 0.0));

    // Collect counts from target network
    for (int d = 0; d < tgt_M; ++d) {
        for (int n = 0; n < tgt_N[d]; ++n) {
            int cur_topic = assign_t[d][n];
            int cur_word = text_network.tgt_blobs[d][n];
            tmp_counts[cur_topic][cur_word] += 1.0;
        }
    }

    // Collect counts from source network
    for (int d = 0; d < src_M; ++d) {
        for (int n = 0; n < src_N[d]; ++n) {
            int cur_topic = assign_t_[d][n];
            int cur_word = text_network.src_blobs[d][n]; 
            tmp_counts[cur_topic][cur_word] += 1.0;
        }
    }

    // Compute phi
    for (int t = 0; t < k; ++t) {
        for (int w = 0; w < V; ++w) {
            // double sum_val = 0.0;
            double numerator = tmp_counts[t][w] + alpha_phi;
            /*double denominator = 0.0;
            for (int j = 0; j < V; ++j) {
                denominator += tmp_counts[t][j][iter - num_warmup];
            }
            denominator += V * alpha_phi;*/
            phi[t][w] = numerator;
            // / denominator;
        }
    }
    return phi;
}

std::vector<std::vector<double>> CollapsedGibbsSocLDA::recover_theta() {
    std::vector<std::vector<double>> theta(src_M, std::vector<double>(k, 0.0));
    std::vector<std::vector<double>> tmp_counts(src_M, std::vector<double>(k, 0.0));

    // Collect counts from source network
    for (int d = 0; d < src_M; ++d) {
        for (int n = 0; n < src_N[d]; ++n) {
            int cur_topic = assign_t_[d][n];
            tmp_counts[d][cur_topic] += 1.0;
        }
    }

    // Collect counts from target network, considering connections
    for (int d = 0; d < tgt_M; ++d) {
        for (int n = 0; n < tgt_N[d]; ++n) {
            int cur_topic = assign_t[d][n];
            int cur_c = assign_c[d][n];
            if (cur_c != src_L) {  // Ensure valid source index
                tmp_counts[cur_c][cur_topic] += 1.0;
            }
        }
    }

    // Compute theta
    for (int d = 0; d < src_M; ++d) {
        for (int t = 0; t < k; ++t) {
            // double sum_val = 0.0;
            double numerator = tmp_counts[d][t] + alpha_theta;
            /*double denominator = 0.0;
            for (int j = 0; j < k; ++j) {
                denominator += tmp_counts[d][j][iter - num_warmup];
            }
            denominator += alpha_theta * k;*/
            theta[d][t] = numerator;
            // / denominator;
        }
    }
    return theta;
}

std::vector<std::vector<double>> CollapsedGibbsSocLDA::recover_lambda() {
    std::vector<std::vector<double>> lambdas(tgt_L, std::vector<double>(2, 0.0));
    std::vector<std::vector<double>> tmp_counts(tgt_L, std::vector<double>(2, 0.0));

    // Collect counts from target network
    for (int d = 0; d < tgt_M; ++d) {
        int subreddit = text_network.tgt_subreddits[d];  // Get subreddit index
        for (int n = 0; n < tgt_N[d]; ++n) {
            int cur_s = assign_s[d][n];
            tmp_counts[subreddit][cur_s] += 1.0;
        }
    }

    // Compute lambda values
    for (int r = 0; r < tgt_L; ++r) {
        double sum_cite = 0;
        double sum_inno = 0;
        /*double denom = 0.0;
        for (int j = 0; j < 2; ++j) {
            denom += tmp_counts[r][j][iter - num_warmup];
        }
        denom -= forced_innovation_count[r];
        denom += lambda_theta + lambda_psi;*/
        lambdas[r][0] += (tmp_counts[r][0] + lambda_theta);
        // / denom;
        lambdas[r][1] += (tmp_counts[r][1] - forced_innovation_count[r] + lambda_psi);
        // / denom;
    }
    return lambdas;
}

// Initialize the Gibbs sampler
void CollapsedGibbsSocLDA::init_gibbs(int n_gibbs) {
    
    // Resize assignment matrices
    assign_c.resize(tgt_M);
    assign_s.resize(tgt_M);
    assign_t.resize(tgt_M);
    for (int d = 0; d < tgt_M; ++d) {
        assign_c[d] = std::vector<int>(tgt_N[d], 0);
        assign_s[d] = std::vector<int>(tgt_N[d], 0);
        assign_t[d] = std::vector<int>(tgt_N[d], 0);
    }

    assign_t_.resize(src_M);
    for (int d = 0; d < src_M; ++d) {
        assign_t_[d] = std::vector<int>(src_N[d], 0);
    }

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
    std::cout << "Init'ing src vals" << std::endl;
    for (int d = 0; d < src_M; ++d) {
        for (int n = 0; n < src_N[d]; ++n) {
            int w_dn = text_network.src_blobs[d][n];
            int cur_topic = topic_dist(gen);
            assign_t_[d][n] = cur_topic;

            // Increment counters
            wt[w_dn][cur_topic]++;
            c_t_[d][cur_topic]++;
            t_sum[cur_topic]++;
        }
    }

    std::cout << "Init'ing tgt vals" << std::endl;
    // Initialize values for each tgt comment
    for (int d = 0; d < tgt_M; ++d) {
        int r = text_network.tgt_subreddits[d];
        for (int n = 0; n < tgt_N[d]; ++n) {
            int w_dn = text_network.tgt_blobs[d][n];

            // Assign innovation flag (s)
            if (text_network.edges[d].empty()) {
                assign_s[d][n] = 1;
                forced_innovation_count[r]++;
            } else {
                assign_s[d][n] = binary_dist(gen);
            }

            // Assign source subreddit (c)
            if (assign_s[d][n] == 0) {
                std::uniform_int_distribution<int> edge_dist(0, text_network.edges[d].size() - 1);
                assign_c[d][n] = text_network.edges[d][edge_dist(gen)];
            } else {
                assign_c[d][n] = src_L;
            }

            // Assign topic (t)
            assign_t[d][n] = topic_dist(gen);

            // Increment counters
            int cur_t = assign_t[d][n];
            int cur_s = assign_s[d][n];
            int cur_c = assign_c[d][n];

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
    for (int r = 0; r < tgt_L; r++) {
        std::cout << forced_innovation_count[r] << std::endl;
    }

    std::cout << "Writing params" << std::endl;
    init3D(output_dir + "/gamma.txt", n_gibbs + 1, tgt_M, tgt_N);
    init3D(output_dir + "/psi.txt", n_gibbs + 1, tgt_M, tgt_N);
    init3D(output_dir + "/phi.txt", n_gibbs + 1, tgt_M, tgt_N);
    init3D(output_dir + "/theta.txt", n_gibbs + 1, src_M, src_N);
    init3D(output_dir + "/lambda.txt", n_gibbs + 1, src_M, src_N);
}

std::vector<double> CollapsedGibbsSocLDA::conditional_prob_cs(int w_dn, int d, int r, int t, bool print) {
    size_t edge_count = text_network.edges[d].size();
    std::vector<double> prob(edge_count + 1, 0.0);

    for (size_t ind = 0; ind < edge_count; ind++) {
        int i = text_network.edges[d][ind];

        double _1 = (c_t_[i][t] + ct[i][t] + alpha_theta) / (src_N[i] + c_sum[i] + k * alpha_theta);
        double _2 = (dc[d][i] + (alpha_sum_edges/edge_count)) / (d_cited_sum[d] + edge_count * (alpha_sum_edges/edge_count));
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


void CollapsedGibbsSocLDA::update_cs(int d, int n, int r) {
    if (text_network.edges[d].empty()) {
        // assign_c[d][n][cs_iter + 1] = assign_c[d][n][cs_iter];
        // assign_s[d][n][cs_iter + 1] = assign_s[d][n][cs_iter];
        return;
    }

    int w_dn = text_network.tgt_blobs[d][n];
    const std::vector<int>& edges = text_network.edges[d];

    int i_t = assign_t[d][n];
    int i_c = assign_c[d][n];
    int i_s = assign_s[d][n];
    /*if (d == 0 && n == 0) {
        std::cout << "Current c: " << i_c << " Current s: " << i_s << std::endl;
    }*/
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
    std::vector<double> prob = conditional_prob_cs(w_dn, d, r, i_t, d==0 && n==0);
    
    /*if (d == 0 && n == 0) {
        for (int counter = 0; counter < edges.size()+1; counter++ ) {
            std::cout << prob[counter] << " "; 
        }
        std::cout << std::endl;
    }*/
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

    assign_c[d][n] = new_c;
    assign_s[d][n] = new_s;
}

void CollapsedGibbsSocLDA::update_t(int d, int n, int r) {
    int w_dn = text_network.tgt_blobs[d][n];

    int i_t = assign_t[d][n];
    int i_c = assign_c[d][n];
    int i_s = assign_s[d][n];

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

    assign_t[d][n] = i_tp1;
}

void CollapsedGibbsSocLDA::update_t_(int c_, int n) {
    int w_dn = text_network.src_blobs[c_][n];
    int i_t_ = assign_t_[c_][n];

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

    assign_t_[c_][n] = i_tp1;
}
