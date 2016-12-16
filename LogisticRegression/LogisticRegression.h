//
// Created by ustczmk on 16/12/10.
//
#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace std;

#define INPUT_FEATURES 2
#define OUTPUT_NUM 1
#define LEARNING_RATE 0.8
#define REGULAR_LAMBDA 1e-7

typedef struct TrainNode {
    vector<double> weight;
    double value;
} TrainNode;

typedef struct OutputNode {
    double out_val, real_val, bias_derivative, bias;
    vector<double> derivative_sum;
} OutputNode;

typedef struct DataGroup {
    vector<double> in, out;
} DataGroup;

inline double genRandom() {
    return ((2.0 * (double)rand() / RAND_MAX) - 1);
}

inline double sigmoid(double x) {
    double ans = 1 / (1 + exp(-x));
    return ans;
}

class LogisticRegression {
public:
    LogisticRegression();
    void forwardPropagation();
    void train(vector<DataGroup> train_set, double threshold);
    void predict(DataGroup& test_input);
    void setInputOutput(DataGroup output);
    TrainNode* input_node[INPUT_FEATURES];
    OutputNode* output_node[OUTPUT_NUM];
    double error;
};
