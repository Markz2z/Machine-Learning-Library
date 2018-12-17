//
// Created by ustczmk on 17/2/14.
//
#include "NeuralNetworkBP.h"
#include <sstream>

NeuralNetwork::NeuralNetwork(int input_nodes, const std::vector<int>& hidden_layer_param,
        int output_nodes) : _input_layer(input_nodes), _output_layer(output_nodes),
        _hidden_layers(hidden_layer_param.size()), _loss(100.f) {
    srand((unsigned)time(NULL));
    int hidden_layers = hidden_layer_param.size();
    //init input layer
    for (auto& node : _input_layer) {
        int first_hidden_nodes = _hidden_layers[0].size();
        node.weight.resize(first_hidden_nodes);
        node.weight_derivative_sum.resize(first_hidden_nodes, 0.f);
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
            hidden_node.weight_derivative_sum.resize(prev_layer_nodes, 0.f);
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

    const auto& first_hidden_layer = _hidden_layers[0];
    for (auto& node : _input_layer) {
        for (int j = 0; j < first_hidden_layer.size(); ++j) {
            node.weight_derivative_sum[j] += first_hidden_layer[j].derivative * node.value;
        }
    }

    for (int layer_idx = 0; layer_idx < _hidden_layers.size(); ++layer_idx) {
        auto& layer = _hidden_layers[layer_idx];
        for (auto& node : layer) {
            node.bias_derivative_sum += node.derivative;
            const auto& next_layer = layer_idx == _hidden_layers.size() - 1 ? _output_layer : _hidden_layers[layer_idx + 1];
            int next_idx = 0;
            for (const auto& next_node : next_layer) {
                node.weight_derivative_sum[next_idx++] += next_node.derivative * node.value;
            }
        }
    }

    for (auto& output_node : _output_layer) {
        output_node.bias_derivative_sum += output_node.derivative;
    }
}

void NeuralNetwork::predict(DataGroup& test_input) {
    std::stringstream ss;
    for (int i = 0; i < _input_layer.size(); ++i) {
        ss << test_input.in[i] << "\n";
    }
    initialize(test_input);
    feed_forward();
    ss << " ====== \n";
    for (int i = 0; i < _output_layer.size(); ++i) {
        test_input.out[i] = _output_layer[i].value;
        ss << _output_layer[i].value << "\n";
    }
    std::cout << ss.str() << std::endl;
}

void NeuralNetwork::initialize(const DataGroup& data) {
    for (int i = 0; i < _input_layer.size(); ++i) {
        _input_layer[i].value = data.in[i];
    }
    for (int i = 0; i < _output_layer.size(); ++i) {
        _output_layer[i].real_val = data.out[i];
    }
}

/*
 * using for debug to dump all of the parameters in neural network
 * */
void NeuralNetwork::print_all_node() {
    std::stringstream ss;
    int idx = 0;
    for (const auto& node : _input_layer) {
        ss << "inputNode[" << idx++ << "]:" << node.value << " ";
    }
    ss << "\n";

    int layer_idx = 0;
    for (const auto& layer : _hidden_layers) {
        int node_idx = 0;
        for (const auto& node : layer) {
            ss << "layer[" << layer_idx << "] node[" << node_idx++ << "] : " << node.value << " ";
        }
        ss << "\n";
        ++layer_idx;
    }

    idx = 0;
    for (const auto& node : _output_layer) {
        ss << "outputNode[" << idx++ << "] : " << node.value << " ";
    }
    ss << "\n";
    std::cout << ss.str() << std::endl;
}

void NeuralNetwork::train(const std::vector<DataGroup>& train_set, double threshold) {
    int layer, idx;
    std::size_t train_size = train_set.size();
    while(_loss > threshold) {
        _loss = 0.f;

        //1. init
        for (auto& input_node : _input_layer) {
            input_node.weight_derivative_sum.assign(_hidden_layers[0].size(), 0.f);
        }

        for (int i = 0; i < _hidden_layers.size(); ++i) {
            auto& layer = _hidden_layers[i];
            int next_layer_nodes = i == _hidden_layers.size() ? _output_layer.size() : _hidden_layers[i+1].size();
            for (auto& node : layer) {
                node.weight_derivative_sum.assign(next_layer_nodes, 0.f);
                node.bias_derivative_sum = 0.f;
            }
        }

        for (auto& output_node : _output_layer) {
            output_node.bias_derivative_sum = 0.f;
        }

        //2. train data
        for (const auto& data : train_set) {
            initialize(data);
            feed_forward();
            back_propagation();
            //print_all_node();
        }


        //3. update weight and bias
        for (auto& input_node : _input_layer) {
            for (int j = 0; j < _hidden_layers[0].size(); ++j) {
                input_node.weight[j] -= LEARNING_RATE * input_node.weight_derivative_sum[j] / train_size;
            }
        }

        for (int layer_idx = 0; layer_idx < _hidden_layers.size(); ++layer_idx) {
            auto& layer = _hidden_layers[layer_idx];
            for (auto& node : layer) {
                node.bias -= LEARNING_RATE * node.bias_derivative_sum / train_size;
                int next_layer_size = layer_idx == _hidden_layers.size() - 1 ? _output_layer.size() : _hidden_layers[layer_idx+1].size();
                for (int next_idx = 0; next_idx < next_layer_size; ++next_idx) {
                    node.weight[next_idx] -= LEARNING_RATE * node.weight_derivative_sum[next_idx] / train_size;
                }
            }
        }

        for (auto& output_node : _output_layer) {
            output_node.bias -= LEARNING_RATE * output_node.bias_derivative_sum / train_size;
        }

        _loss /= train_size;
        std::cout << "error : " << _loss << std::endl;
    }
}
