//
// Created by ustczmk on 17/2/14.
//
#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define LEARNING_RATE 0.9

double genRandom() {
    return ((2.0 * (double)rand() / RAND_MAX) - 1);
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

typedef struct NeuronNode {
    NeuronNode() : bias(genRandom()) {}
    std::vector<double> weight, weight_derivative_sum;
    double real_val, value, bias, derivative, bias_derivative_sum;
} NeuronNode;

typedef struct InputNode {
    std::vector<double> weight, weight_derivative_sum;
    double value;
} InputNode;

typedef struct DataGroup {
    std::vector<double> in, out;
} DataGroup;

class NeuralNetwork {
public:
    NeuralNetwork(int input_nodes, const std::vector<int>& hidden_layer_param, int output_nodes);
    void feed_forward();
    void back_propagation();
    void train(const std::vector<DataGroup>& train_set, double threshold);
    void predict(DataGroup& test_input);
    void print_all_node();
    void initialize(const DataGroup& output);

private:
    std::vector<InputNode> _input_layer;
    std::vector<std::vector<NeuronNode>> _hidden_layers;
    std::vector<NeuronNode> _output_layer;
    double _loss;
};
