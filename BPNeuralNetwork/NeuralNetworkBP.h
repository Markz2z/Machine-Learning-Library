//
// Created by ustczmk on 17/2/14.
//
#pragma once

#include <cassert>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define LEARNING_RATE 0.9
#define DEBUG_LEVEL 0

static double genRandom() {
    return ((2.0 * (double)rand() / RAND_MAX) - 1);
}

static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

typedef std::vector<double> Tensor;

/**
 * @param
 * weight : neural network weight
 * batch_weight_derivatives : sum up batch of data's derivatives
 * real_val : trainning real output value
 * value : neuron node value before activation
 * bias : neuron network bias
 * derivative : neuron node derivative, useful for compute previous layer's derivative
 * batch_bias_derivative : sum up batch of data bias
 *
 * */

struct NeuronNode {
    NeuronNode() : bias(genRandom()) {}
    Tensor weight, batch_weight_derivatives;
    double real_val, value, bias, derivative, batch_bias_derivative;
};

struct InputNode {
    Tensor weight, batch_weight_derivatives;
    double value;
};

/**
 * a alice of training data, input and output
 *
 * */

struct DataGroup {
    Tensor in;
    mutable Tensor out;
};

class NeuralNetwork {
public:
    NeuralNetwork(int input_nodes, const std::vector<int>& hidden_layer_param, int output_nodes);
    void add_train_slice(std::vector<double>&& in, std::vector<double>&& out);
    void train(double threshold);
    void predict(const Tensor& input);

    void clear_batch() {
        _batch_size = 0;
        _train_set.clear();
    }

private:
    void update_param();
    void initialize();
    void set_train_data(const DataGroup& output);
    void set_input_data(const Tensor& in);
    void feed_forward();
    void back_propagation();
    void print_all_node() const;
    void dump_input_output() const;
 
    std::vector<InputNode> _input_layer;
    std::vector<std::vector<NeuronNode>> _hidden_layers;
    std::vector<NeuronNode> _output_layer;
    double _loss;
    std::size_t _batch_size;
    std::vector<DataGroup> _train_set;
};
