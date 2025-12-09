#include "utils.h"
#include "dataset.h"
#include "model.h"
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {

    //Parse command line arguments
    std::string src_blobs_file(argv[1]);
    std::string tgt_blobs_file(argv[2]);
    std::string edges_file(argv[3]);
    std::string subreddits_file(argv[4]);
    std::string output_dir(argv[5]);

    //Read dataset in from file
    std::vector<std::vector<int>> src_blobs = read2D(src_blobs_file);
    std::vector<std::vector<int>> tgt_blobs = read2D(tgt_blobs_file);
    std::vector<int> tgt_subreddits = read1D(subreddits_file);  
    std::vector<std::vector<int>> edges = read2D(edges_file);
    
    //zeros out all edges if you want to run vanilla LDA
   //std::vector<std::vector<int>> edges;
   // for (int i = 0; i < tgt_subreddits.size(); i ++ ) {
//	std::vector<int> empty;
//	edges.push_back(empty);
  //  }

    int iterations = 1000;
    int warmup = 1;

    int num_src_subreddits = src_blobs.size();
    
    //deduce number of target subreddits from subreddits vector
    //num subreddits is 1 larger than the largest subreddit index
    int num_tgt_subreddits = *max_element(tgt_subreddits.begin(), tgt_subreddits.end()) + 1;
    
    //deduce number of target subreddits src_blobs and tgt_blobs
    int vocab_size = -1;
    int cur_row = 0;
    for (const auto& blob : src_blobs) {
        if (!blob.empty()) {
            int row_max = *max_element(blob.begin(), blob.end());
            vocab_size = (vocab_size > row_max) ? vocab_size : row_max;
        }
    }
    for (const auto& blob : tgt_blobs) {
        if (!blob.empty()) {
            int row_max = *max_element(blob.begin(), blob.end());
            vocab_size = (vocab_size > row_max) ? vocab_size : row_max;
        }
    }
    //vocab size is 1 larger than the largest word index
    vocab_size++;

    const TextNetwork text_network = {src_blobs, tgt_blobs, edges, tgt_subreddits, vocab_size, num_src_subreddits, num_tgt_subreddits};    
    std::cout << "Made Text network" << std::endl;

    //Initialize model
    CollapsedGibbsSocLDA model(text_network, 50);
    std::cout << "Initalized" << std::endl;

    //Run Gibbs sampler
    model.run_gibbs(iterations, true);

    //Recover parameters
    std::vector<std::vector<std::vector<double>>> gamma = model.recover_gamma(iterations, warmup);
    std::vector<std::vector<std::vector<double>>> psi = model.recover_psi(iterations, warmup);
    std::vector<std::vector<std::vector<double>>> phi = model.recover_phi(iterations, warmup);
    std::vector<std::vector<std::vector<double>>> theta = model.recover_theta(iterations, warmup);
    std::vector<std::vector<std::vector<double>>> lambda = model.recover_lambda(iterations, warmup);

    //Save to parameters to output file   
    write3D(output_dir + "/gamma.txt", gamma);
    write3D(output_dir + "/psi.txt", psi);
    write3D(output_dir + "/phi.txt", phi);
    write3D(output_dir + "/theta.txt", theta);
    write3D(output_dir + "/lambda.txt", lambda);

}
