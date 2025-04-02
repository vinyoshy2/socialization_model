#ifndef MODEL
#define MODEL

#include <vector>
#include "dataset.h"
#include <random>

class CollapsedGibbsSocLDA {

public:

    TextNetwork text_network;
    int V;
    int k;

    std::vector<int> src_N;
    std::vector<int> tgt_N;
    int src_M;
    int tgt_M;
    int src_L;
    int tgt_L;

    // Hyperparameters
    double alpha_phi;
    double alpha_theta;
    double alpha_gamma;
    double alpha_psi;
    double lambda_theta;
    double lambda_psi;

    // Count matrices
    std::vector<std::vector<int>> dc;
    std::vector<std::vector<int>> ct;
    std::vector<std::vector<std::vector<int>>> rts;
    std::vector<std::vector<int>> c_t_;
    std::vector<std::vector<int>> wt;
    std::vector<int> forced_innovation_count;

    // Row/Column sums for count matrices
    std::vector<int> c_sum;
    std::vector<int> d_cited_sum;
    std::vector<int> r0_sum;
    std::vector<int> r1_sum;
    std::vector<int> t_sum;

    // Gibbs sampler assignment matrices
    std::vector<std::vector<std::vector<int>>> assign_c;
    std::vector<std::vector<std::vector<int>>> assign_s;
    std::vector<std::vector<std::vector<int>>> assign_t;
    std::vector<std::vector<std::vector<int>>> assign_t_;


    // Random number generator
    std::random_device rd;
    std::mt19937 gen;

    // Constructor
    CollapsedGibbsSocLDA(const TextNetwork& text_network, int n_topic); 

    // Gibbs sampling function
    void run_gibbs(int n_gibbs, bool verbose);
    
    // Functions for recovering parameters
    std::vector<std::vector<std::vector<double>>> recover_gamma(int total_iter, int num_warmup);
    std::vector<std::vector<std::vector<double>>> recover_psi(int total_iter, int num_warmup);
    std::vector<std::vector<std::vector<double>>> recover_phi(int total_iter, int num_warmup);
    std::vector<std::vector<std::vector<double>>> recover_theta(int total_iter, int num_warmup);
    std::vector<std::vector<std::vector<double>>> recover_lambda(int total_iter, int num_warmup);

private:

    void init_gibbs(int n_gibbs);

    std::vector<double> conditional_prob_cs(int w_dn, int d, int r, int t);

    std::vector<double> conditional_prob_t(int w_dn, int d, int r, int c, int s);

    std::vector<double> conditional_prob_t_(int w_c_n, int c_);

    void update_cs(int d, int n, int r, int cs_iter, int t_iter);

    void update_t(int d, int n, int r, int cs_iter, int t_iter);

    void update_t_(int c_, int n, int t_iter);
};

#endif