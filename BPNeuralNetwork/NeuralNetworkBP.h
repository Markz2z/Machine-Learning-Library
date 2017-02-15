//
// Created by ustczmk on 17/2/14.
//
#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace std;

#define INPUT_LAYER_NODES 2
#define HIDE_LAYERS 5
#define HIDE_LAYER_NODES 7
#define OUTPUT_LAYER_NODES 1
#define LEARNING_RATE 0.9

typedef struct NeuronNode {
    vector<double> weight, weight_derivative_sum;
    double value, bias, derivative, bias_derivative_sum;
} NeuronNode;

typedef struct InputNode {
    vector<double> weight, weight_derivative_sum;
    double value;
} InputNode;

typedef struct OutputNode {
    double compute_val, real_val, bias, derivative, bias_derivative_sum;
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

class NeuralNetwork {
public:
    NeuralNetwork();
    void forwardPropagation();
    void backPropagation();
    void train(vector<DataGroup> train_set, double threshold);
    void predict(DataGroup& test_input);
    void printAllNode();
    void setInputOutput(DataGroup output);
    InputNode* input_node[INPUT_LAYER_NODES];
    vector<NeuronNode*> hide_node[HIDE_LAYERS];
    OutputNode* output_node[OUTPUT_LAYER_NODES];
    double error;
};