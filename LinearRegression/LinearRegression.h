//
// Created by ustczmk on 16/12/14.
//

#ifndef LINEARREGRESSION_LINEARREGRESSION_H
#define LINEARREGRESSION_LINEARREGRESSION_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace std;

#define INPUT_FEATURES 2
#define OUTPUT_NUM 1
#define LEARNING_RATE 0.005
//#define DEBUG 1

typedef struct InputNode {
    vector<double> weight;
    double value;
} InputNode;

typedef struct OutputNode {
    double out_val, real_val, bias, bias_derivative;
    vector<double> derivative_sum;
} OutputNode;

typedef struct DataGroup {
    vector<double> in, out;
} DataGroup;

inline double genRandom() {
    return ((2.0 * (double)rand() / RAND_MAX) - 1);
}

class LinearRegression {
public:
    LinearRegression();
    void forwardPropagation();
    void train(vector<DataGroup> train_set, double threshold);
    void predict(DataGroup& test_input);
    void setInputOutput(DataGroup output);
    InputNode* input_node[INPUT_FEATURES];
    OutputNode* output_node[OUTPUT_NUM];
    double error;
};

#endif //LINEARREGRESSION_LINEARREGRESSION_H