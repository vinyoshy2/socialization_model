#include "utils.h"
#include "dataset.h"
#include "model.h"
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

struct Options {
    std::string input_dir;
    std::string output_dir;
    int topics = 50;
    int iterations = 1000;
    int warmup_steps = 200;
    float alpha_sum_topics = 1.0f;
    float alpha_sum_vocab = 1.0f;
    float alpha_sum_edges = 1.0f;
};

Options parseArgs(int argc, char** argv) {
    Options opt;
    std::vector<std::string> args(argv + 1, argv + argc);

    int positionalCount = 0;

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];

        // --------------------
        // Positional arguments
        // --------------------
        if (a[0] != '-') {
            if (positionalCount == 0) {
                opt.input_dir = a;
            } else if (positionalCount == 1) {
                opt.output_dir = a;
            } else {
                throw std::runtime_error("Unexpected positional argument: " + a);
            }
            positionalCount++;
            continue;
        }

        // --------------------
        // Flags with values
        // --------------------
        auto require_value = [&](const std::string& flag) {
            if (i + 1 >= args.size()) {
                throw std::runtime_error("Missing value for " + flag);
            }
            return args[++i];
        };

        if (a == "--topics") {
            opt.topics = std::stoi(require_value(a));
        }
        else if (a == "--iters") {
            opt.iterations = std::stoi(require_value(a));
        }
        else if (a == "--warmup") {
            opt.warmup_steps = std::stoi(require_value(a));
        }
        else if (a == "--alpha-vocab") {
            opt.alpha_sum_vocab = std::stof(require_value(a));
        }
        else if (a == "--alpha-topics") {
            opt.alpha_sum_topics = std::stof(require_value(a));
        }
        else if (a == "--alpha-edges") {
            opt.alpha_sum_edges = std::stof(require_value(a));
        }
        else {
            throw std::runtime_error("Unknown flag: " + a);
        }
    }

    if (positionalCount < 2) {
        throw std::runtime_error("Usage: program <input_dir> <output_dir> [--topics N] [--iters N] [--warmup N] [--alpha-topics F] [--alpha-vocab F] [--alpha-edges F]");
    }

    return opt;
}

int main(int argc, char** argv) {
    
    try {
        Options opt = parseArgs(argc, argv);

        std::cout << "input_dir:    " << opt.input_dir << "\n";
        std::cout << "output_dir:   " << opt.output_dir << "\n";
        std::cout << "topics:       " << opt.topics << "\n";
        std::cout << "iterations:   " << opt.iterations << "\n";
        std::cout << "warmup_steps: " << opt.warmup_steps << "\n";
        std::cout << "alpha_sum_vocab:    " << opt.alpha_sum_vocab << "\n";
        std::cout << "alpha_sum_topics:    " << opt.alpha_sum_topics << "\n";
        std::cout << "alpha_sum_edges:    " << opt.alpha_sum_edges << "\n";
    
    	//Parse command line arguments
        std::string src_blobs_file = opt.input_dir + "/src_blobs.txt";
        std::string tgt_blobs_file = opt.input_dir + "/tgt_blobs.txt";
        std::string edges_file = opt.input_dir + "/edges.txt";
        std::string subreddits_file = opt.input_dir + "/subreddits.txt";
        std::string output_dir = opt.output_dir;

        //Read dataset in from file
        std::vector<std::vector<int>> src_blobs = read2D(src_blobs_file);
        std::vector<std::vector<int>> tgt_blobs = read2D(tgt_blobs_file);
        std::vector<int> tgt_subreddits = read1D(subreddits_file);  
        std::vector<std::vector<int>> edges = read2D(edges_file);

        int iterations = opt.iterations;
        int warmup = opt.warmup_steps;

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
        CollapsedGibbsSocLDA model(text_network, opt.topics, opt.alpha_sum_topics, opt.alpha_sum_vocab, opt.alpha_sum_edges);
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
        write3D(output_dir + "/assign_c.txt", model.assign_c);

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
