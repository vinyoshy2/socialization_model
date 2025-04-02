#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include "utils.h"

int weighted_sample(const std::vector<double>& prob, std::mt19937& gen) {
    std::discrete_distribution<int> dist(prob.begin(), prob.end());
    return dist(gen);
}


std::vector<int> read1D(std::string filename) {
    // Open the file for reading
    std::ifstream fin(filename);
 
    // Create an empty vector
    std::vector<int> v;
    int size;
    fin >> size;
    v.resize(size);
 
    // Read the contents of the file and store them in the
    // vector
    int c;
    int pos {0};
    while (fin >> c) {
        v[pos] = c;
        ++pos;
    }
 
    // Close the file
    fin.close();

    return v;
}

std::vector<std::vector<int>> read2D(std::string filename) {
    // Open the file for reading
    std::ifstream fin(filename);
 
    // Create an empty vector of vectors
    std::vector<std::vector<int>> v;
    int num_rows;

    // First line of the file will contain number of rows
    fin >> num_rows;
    v.resize(num_rows);
    
    int row_sizes[num_rows];

    // Next num_rows lines will contain number of elements in each row
    // Note that this is a "ragged" 2D array -- rows have diff. # of elements
    int row_size;
    for (int i = 0; i < num_rows; i++) {
        fin >> row_size;
        v[i].resize(row_size);
        row_sizes[i] = row_size;
    }
 
    // Subsequent lines will contain the elements of the matrix
    int c;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < row_sizes[i]; j++) {
            fin >> c;
            v[i][j] = c;
        }
    }
 
    // Close the file
    fin.close();
    return v;
}

void write1D(std::string filename, std::vector<double> v) {
    std::ofstream outputFile;
    outputFile.open(filename);
    
    //1D vector Files begin with number of elements in the vector
    outputFile << v.size() << std::endl;
    
    for (int i = 0; i < v.size(); i++) {
        outputFile << v[i] << " "; //Elements separated by spaces
    }
    
    outputFile << std::endl;
    outputFile.close();
}

void write2D(std::string filename, std::vector<std::vector<double>> v) {
    std::ofstream outputFile;
    outputFile.open(filename);
    
    //2D vector files begin with number of rows
    outputFile << v.size() << std::endl;
    
    //2D vectors files then contain number of elements in each row
    for (int i = 0; i < v.size(); i++) {
        outputFile << v[i].size() << std::endl;
    }

    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            outputFile << v[i][j] << " "; //Row elements separated by spaces

        }
        outputFile << std::endl;; //Rows separated by newlines
    }

    outputFile.close();

}

void write3D(std::string filename, std::vector<std::vector<std::vector<double>>> v) {
    std::ofstream outputFile;
    outputFile.open(filename);
    
    //2D vector files begin with number of rows
    outputFile << v.size() << std::endl;
    
    //2D vectors files then contain number of elements in each row
    for (int i = 0; i < v.size(); i++) {
        outputFile << v[i].size() << std::endl;
    }

    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            outputFile << v[i][j].size() << std::endl;
        }
    }

    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            for (int k = 0; k < v[i][j].size(); k++) {
                outputFile << v[i][j][k] << " "; //Row elements separated by spaces
            }
            outputFile << std::endl; //Rows separated by newlines
        }
        outputFile << std::endl << std::endl; //Rows separated by newlines
    }
    outputFile.close();
}
