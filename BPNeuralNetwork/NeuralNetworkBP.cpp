//
// Created by ustczmk on 17/2/14.
//
#include "NeuralNetworkBP.h"
#include <sstream>

void NeuralNetwork::add_train_slice(std::vector<double>&& in, std::vector<double>&& out) {
    assert("in size != input layer nodes" && in.size() == _input_layer.size());
    assert("output size != output layer nodes" && out.size() == _output_layer.size());

    _train_set.emplace_back();
    _train_set.back().in = in;
    _train_set.back().out = out;
}

NeuralNetwork::NeuralNetwork(int input_nodes, const std::vector<int>& hidden_layer_param,
        int output_nodes) : _input_layer(input_nodes), _output_layer(output_nodes),
        _hidden_layers(hidden_layer_param.size()), _loss(100.f) {
    srand((unsigned)time(NULL));
    int hidden_layers = hidden_layer_param.size();
    //init input layer
    for (auto& node : _input_layer) {
        int first_hidden_nodes = _hidden_layers[0].size();
        node.weight.resize(first_hidden_nodes);
        node.batch_weight_derivatives.resize(first_hidden_nodes, 0.f);
        for (int j = 0; j < first_hidden_nodes; ++j) {
            node.weight[j] = genRandom();
        }
    }

    //init hidden layer
    int hidden_layer_idx = 0;
    for (auto& hidden_layer : _hidden_layers) {
        hidden_layer.resize(hidden_layer_param[hidden_layer_idx]);
        for (auto& hidden_node : hidden_layer) {
            int next_layer_nodes = hidden_layer_idx == hidden_layers - 1 ?
                output_nodes : hidden_layer_param[hidden_layer_idx + 1];
            hidden_node.weight.resize(next_layer_nodes);
            for (int k = 0; k < next_layer_nodes; ++k) {
                hidden_node.weight[k] = genRandom();
            }
            int prev_layer_nodes = hidden_layer_idx == 0 ? input_nodes : hidden_layer_param[hidden_layer_idx - 1];
            hidden_node.batch_weight_derivatives.resize(prev_layer_nodes, 0.f);
        }
        ++hidden_layer_idx;
    }
}

void NeuralNetwork::feed_forward() {
    // hidden layer
    for (int layer_idx = 0; layer_idx < _hidden_layers.size(); ++layer_idx) {
        auto& layer = _hidden_layers[layer_idx];
        for (int node_idx = 0; node_idx < layer.size(); ++node_idx) {
            auto& node = layer[node_idx];
            double sum = node.bias;
            if (layer_idx == 0) {
                for (const auto& input_node : _input_layer) {
                    sum += input_node.value * input_node.weight[node_idx];
                }
            } else {
                const auto& prev_layer = _hidden_layers[layer_idx - 1];
                for (const auto& prev_node : prev_layer) {
                    sum += prev_node.value * prev_node.weight[node_idx];
                }
            }
            node.value = sum;
        }
    }

    // output layer
    for (int node_idx = 0; node_idx < _output_layer.size(); ++node_idx) {
        auto& output_node = _output_layer[node_idx];
        double sum = output_node.bias;
        const auto& hide_layer = _hidden_layers.back();
        for (const auto& hide_node : hide_layer) {
            sum += hide_node.value * hide_node.weight[node_idx];
        }
        output_node.value = sigmoid(sum);
    }
}

void NeuralNetwork::back_propagation() {
    for (auto& output_node : _output_layer) {
        double delta = output_node.value - output_node.real_val;
        output_node.derivative = delta * output_node.value * (1 - output_node.value);
        _loss += delta * delta / 2;
    }

    for (auto layer = _hidden_layers.rbegin(); layer != _hidden_layers.rend(); ++layer) {
        for (auto& node : *layer) {
            double sigmoid_derivative = node.value * (1 - node.value), sum = 0.f;
            auto next_layer = layer == _hidden_layers.rbegin() ? _output_layer : *(layer - 1);
            int next_layer_idx = 0;
            for (const auto& next_node : next_layer) {
                sum += sigmoid_derivative * next_node.derivative * node.weight[next_layer_idx++];
            }
            node.derivative = sum;
        }
    }

    const auto& first_hidden_layer = _hidden_layers.front();
    for (auto& input_node : _input_layer) {
        int next_node_idx = 0;
        for (const auto& next_node : first_hidden_layer) {
            input_node.batch_weight_derivatives[next_node_idx++] += next_node.derivative * input_node.value;
        }
    }

    for (int layer_idx = 0; layer_idx < _hidden_layers.size(); ++layer_idx) {
        auto& hidden_layer = _hidden_layers[layer_idx];
        for (auto& hidden_node : hidden_layer) {
            hidden_node.batch_bias_derivative = hidden_node.derivative;
            const auto& next_layer = layer_idx == _hidden_layers.size() - 1 ? _output_layer : _hidden_layers[layer_idx + 1];
            int next_idx = 0;
            for (const auto& next_node : next_layer) {
                hidden_node.batch_weight_derivatives[next_idx++] += next_node.derivative * hidden_node.value;
            }
        }
    }

    for (auto& output_node : _output_layer) {
        output_node.batch_bias_derivative += output_node.derivative;
    }
}

void NeuralNetwork::dump_input_output() const {
    if (DEBUG_LEVEL <= 0) return;
    std::stringstream ss;
    ss << "INPUT : ";
    for (const auto& input_node : _input_layer) {
        ss << input_node.value << "\t";
    }
    ss << "\n";

    ss << "OUTPUT : ";
    for (const auto& output_node : _output_layer) {
        ss << output_node.value << "\t";
    }
    ss << "\n";
    std::cout << ss.str() << std::endl;
}

void NeuralNetwork::predict(const Tensor& input) {
    set_input_data(input);
    feed_forward();
    dump_input_output();
}

void NeuralNetwork::set_input_data(const Tensor& in) {
    for (int i = 0; i < _input_layer.size(); ++i) {
        _input_layer[i].value = in[i];
    }
}

void NeuralNetwork::set_train_data(const DataGroup& data) {
    set_input_data(data.in);
    for (int i = 0; i < _output_layer.size(); ++i) {
        _output_layer[i].real_val = data.out[i];
    }
}

/*
 * using for debug to dump all of the parameters in neural network
 */
void NeuralNetwork::print_all_node() const {
    if (DEBUG_LEVEL <= 0) return;
 
    std::stringstream ss;
    int idx = 0;
    for (const auto& node : _input_layer) {
        ss << "inputNode[" << idx++ << "]:" << node.value << "\t";
    }
    ss << "\n";

    int layer_idx = 0;
    for (const auto& layer : _hidden_layers) {
        int node_idx = 0;
        for (const auto& node : layer) {
            ss << "layer[" << layer_idx << "] node[" << node_idx++ << "] : " << node.value << "\t";
        }
        ss << "\n";
        ++layer_idx;
    }

    idx = 0;
    for (const auto& node : _output_layer) {
        ss << "outputNode[" << idx++ << "] : " << node.value << "\t";
    }
    ss << "\n";
    std::cout << ss.str() << std::endl;
}

void NeuralNetwork::initialize() {
    _loss = 0.f;

    for (auto& input_node : _input_layer) {
        input_node.weight.assign(_hidden_layers[0].size(), 0.f);
        input_node.batch_weight_derivatives.assign(_hidden_layers[0].size(), 0.f);
    }

    for (int i = 0; i < _hidden_layers.size(); ++i) {
        auto& layer = _hidden_layers[i];
        int next_layer_nodes = i == (_hidden_layers.size() - 1) ? _output_layer.size() : _hidden_layers[i+1].size();
        for (auto& node : layer) {
            node.batch_weight_derivatives.assign(next_layer_nodes, 0.f);
            node.batch_bias_derivative = 0.f;
        }
    }

    for (auto& output_node : _output_layer) {
        output_node.batch_bias_derivative = 0.f;
    }
}

void NeuralNetwork::update_param() {
    for (auto& input_node : _input_layer) {
        for (int j = 0; j < _hidden_layers[0].size(); ++j) {
            input_node.weight[j] -= LEARNING_RATE * input_node.batch_weight_derivatives[j] / _batch_size;
        }
    }

    for (int layer_idx = 0; layer_idx < _hidden_layers.size(); ++layer_idx) {
        auto& layer = _hidden_layers[layer_idx];
        for (auto& node : layer) {
            node.bias -= LEARNING_RATE * node.batch_bias_derivative / _batch_size;
            int next_layer_size = layer_idx == _hidden_layers.size() - 1 ? _output_layer.size() : _hidden_layers[layer_idx+1].size();
            for (int next_idx = 0; next_idx < next_layer_size; ++next_idx) {
                node.weight[next_idx] -= LEARNING_RATE * node.batch_weight_derivatives[next_idx] / _batch_size;
            }
        }
    }

    for (auto& output_node : _output_layer) {
        output_node.bias -= LEARNING_RATE * output_node.batch_bias_derivative / _batch_size;
    }
    _loss /= _batch_size;
}

void NeuralNetwork::train(double threshold) {
    _batch_size = _train_set.size();
    _loss = 100.0;
    while (_loss > threshold) {
        //1. initialize training environment
        initialize();

        //2. train a batch of  data
        for (const auto& data : _train_set) {
            set_train_data(data);
            feed_forward();
            back_propagation();
            print_all_node();
        }

        //3. update weight and bias
        update_param();
        std::cout << "error : " << _loss << std::endl;
    }
}
