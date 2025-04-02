#include "utils.h"
#include "dataset.h"
#include "model.h"
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {

    // Set up timer
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    //Parse command line arguments
    std::string src_blobs_file(argv[1]);
    std::string tgt_blobs_file(argv[2]);
    std::string edges_file(argv[3]);
    std::string subreddits_file(argv[4]);
    

    //Read dataset in from file
    std::vector<std::vector<int>> src_blobs = read2D(src_blobs_file);
    std::vector<std::vector<int>> tgt_blobs = read2D(tgt_blobs_file);
    std::vector<std::vector<int>> edges = read2D(edges_file);
    std::vector<int> tgt_subreddits = read1D(subreddits_file);  

    int num_src_subreddits = src_blobs.size();
    
    //deduce number of target subreddits from subreddits vector
    //num ubreddits is 1 larger than the largest subreddit index
    int num_tgt_subreddits = *max_element(tgt_subreddits.begin(), tgt_subreddits.end()) + 1;
    
    //deduce number of target subreddits src_blobs and tgt_blobs
    int vocab_size = -1;
    for (const auto& blob : src_blobs) {
        int row_max = *max_element(blob.begin(), blob.end());
        vocab_size = (vocab_size > row_max) ? vocab_size : row_max;
    }
    for (const auto& blob : tgt_blobs) {
        int row_max = *max_element(blob.begin(), blob.end());
        vocab_size = (vocab_size > row_max) ? vocab_size : row_max;
    }
    //vocab size is 1 larger than the largest word index
    vocab_size++;

    const TextNetwork text_network = {src_blobs, tgt_blobs, edges, tgt_subreddits, vocab_size, num_src_subreddits, num_tgt_subreddits};

    //Initialize model
    CollapsedGibbsSocLDA model(text_network, 2);

    //Run Gibbs sampler
    model.run_gibbs(2000, true);

    //Recover parameters
    std::vector<std::vector<std::vector<double>>> gamma = model.recover_gamma(2000, 500);
    std::vector<std::vector<std::vector<double>>> psi = model.recover_psi(2000, 500);
    std::vector<std::vector<std::vector<double>>> phi = model.recover_phi(2000, 500);
    std::vector<std::vector<std::vector<double>>> theta = model.recover_theta(2000, 500);
    std::vector<std::vector<std::vector<double>>> lambda = model.recover_lambda(2000, 500);

    //Save to parameters to output file   
    write3D("results/smallest/gamma.txt", gamma);
    write3D("results/smallest/psi.txt", psi);
    write3D("results/smallest/phi.txt", phi);
    write3D("results/smallest/theta.txt", theta);
    write3D("results/smallest/lambda.txt", lambda);

    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";

}