#ifndef UTILS
#define UTILS

#include <vector>
#include <string>
#include <random>
#include <fstream>


int weighted_sample(const std::vector<double>& prob, std::mt19937& gen);
std::vector<int> read1D(std::string filename);
std::vector<std::vector<int>> read2D(std::string filename);

void write1D(std::string filename, std::vector<double> v);
void write2D(std::string filename, std::vector<std::vector<double>> v);
void write3D(std::string filename, std::vector<std::vector<std::vector<double>>> v);

void init3D(std::string filename, int dim1, int dim2, std::vector<int> dim3);

template <typename T>
void append2D(std::string filename, const std::vector<std::vector<T>>& v) {
    std::ofstream outputFile;
    outputFile.open(filename, std::ios::app);

    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            outputFile << v[i][j] << " ";
        }
        outputFile << std::endl;
    }
    outputFile << std::endl << std::endl;
}

#endif
