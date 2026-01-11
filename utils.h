#ifndef UTILS
#define UTILS

#include <vector>
#include <string>
#include <random>

int weighted_sample(const std::vector<double>& prob, std::mt19937& gen);
std::vector<int> read1D(std::string filename);
std::vector<std::vector<int>> read2D(std::string filename);

void write1D(std::string filename, std::vector<double> v);
void write2D(std::string filename, std::vector<std::vector<double>> v);
void write3D(std::string filename, std::vector<std::vector<std::vector<double>>> v);

void init3D(std::string filename, int dim1, int dim2, std::vector<int> dim3);
void append2D(std::string filename, std::vector<std::vector<double>> v);

#endif
