//
// Created by ustczmk on 17/2/14.
//
#include "NeuralNetworkBP.h"

NeuralNetwork::NeuralNetwork(int input_nodes, const std::vector<int>& hidden_layer_param,
        int output_nodes) {
    srand((unsigned)time(NULL));
    error = 100.f;

    int hidden_layers = hidden_layer_param.size();

    _output_layer.resize(output_nodes);

    //init input layer
    _input_layer.resize(input_nodes);
    for (auto& node : _input_layer) {
        int first_hidden_nodes = hidden_layer_param[0].size();
        node->weight.resize(first_hidden_nodes);
        for (int j = 0; j < node->weight.size(); ++j) {
            node->weight[j] = genRandom();
        }
        node->weight_derivative_sum.resize(first_hidden_nodes, 0.f);
    }

    //init hide layer
    _hidden_layers.resize(hidden_layers);
    int hidden_layer_idx = 0;
    for (auto& hidden_layer : _hidden_layers) {
        hidden_layer.resize(hidden_layer_param[hidden_layer_idx]);
        for (auto& hidden_node : hidden_layer) {
            int weight_cnt = hidden_layer_idx == hidden_layers - 1 ?
                output_nodes : hidden_layer_param[hidden_layer_idx + 1].size();
            hidden_node.weight.resize(weight_cnt);
            for (int k = 0; k < weight_cnt; ++k) {
                hidden_node.weight[k] = genRandom();
            }
            int derivatives = i == 0 ? input_nodes : hidden_layer_param[hidden_layer_idx - 1].size();
            hidden_node.weight_derivative_sum.resize(derivatives, 0.f);
            hidden_node.bias = genRandom();
        }
        ++hidden_layer_idx;
    }

    //init output layer
    _output_layer.resize(output_nodes);
}

void NeuralNetwork::forwardPropagation() {

    // hidden layer
    for (int layer_idx = 0: layer_idx < _hidden_layers.size(); ++layer_idx) {
        auto& layer = _hidden_layers[layer_idx];
        for (int node_idx = 0; node_idx < layer.size(); ++node_idx) {
            auto& node = layer[node_idx];
            double sum = node.bias;
            if (layer_idx == 0) {
                for (const auto& input_node : _input_layer) {
                    sum += input_node.value * input_node.weight[node_idx];
                }
            } else {
                auto& prev_layer = _hidden_layers[layer_idx - 1];
                for (const auto& prev_node : prev_layer) {
                    sum += prev_node.value * prev_node.weight[node_idx];
                }
            }
            node.value = sum;
        }
    }

    for (int node_idx = 0; node_idx < _output_layer.size(); ++node_idx) {
        auto& output_node = _output_layer[node_idx];
        double sum = output_node.bias;
        auto& hide_layer = _hidden_layers.back();
        for (const auto& hide_node : hide_layer) {
            sum += hide_node.value * hide_node.weight[node_idx];
        }
        output_node.compute_val = sigmoid(sum);
    }
}

void NeuralNetwork::backPropagation() {
    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) {
        double t = output_node[i]->compute_val - output_node[i]->real_val;
        error += t * t / 2;
        output_node[i]->derivative = t * output_node[i]->compute_val * (1 - output_node[i]->compute_val);
    }

    for (int i = HIDE_LAYERS - 1; i >= 0; --i) {
        for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
            double sigmoid_derivative = hide_node[i][j]->value * (1 - hide_node[i][j]->value), sum = 0.f;
            if (i==HIDE_LAYERS - 1) {
                for (int k = 0; k < OUTPUT_LAYER_NODES; ++k) {
                    sum += sigmoid_derivative * output_node[k]->derivative * hide_node[i][j]->weight[k];
                }
            } else {
                for (int k = 0; k < HIDE_LAYER_NODES; ++k) {
                    sum += sigmoid_derivative * hide_node[i+1][k]->derivative * hide_node[i][j]->weight[k];
                }
            }
            hide_node[i][j]->derivative = sum;
        }
    }

    for (int i = 0; i < INPUT_LAYER_NODES; ++i) {
        for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
            input_node[i]->weight_derivative_sum[j] += hide_node[0][j]->derivative * input_node[i]->value;
        }
    }

    for (int i = 0; i < HIDE_LAYERS; ++i) {
        for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
            hide_node[i][j]->bias_derivative_sum += hide_node[i][j]->derivative;
            if (i==HIDE_LAYERS-1) {
                for (int k = 0; k < OUTPUT_LAYER_NODES; ++k) {
                    hide_node[i][j]->weight_derivative_sum[k] += output_node[k]->derivative * hide_node[i][j]->value;
                }
            } else {
                for (int k = 0; k < HIDE_LAYER_NODES; ++k) {
                    hide_node[i][j]->weight_derivative_sum[k] += hide_node[i+1][k]->derivative * hide_node[i][j]->value;
                }
            }
        }
    }

    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) {
        output_node[i]->bias_derivative_sum += output_node[i]->derivative;
    }
}

void NeuralNetwork::predict(DataGroup& test_input) {
    for ( int i = 0; i < INPUT_LAYER_NODES; ++i) {
        cout << test_input.in[i] << " ";
    }
    setInputOutput(test_input);
    forwardPropagation();
    cout << " ====> ";
    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) {
        test_input.out[i] = output_node[i]->compute_val;
        cout << output_node[i]->compute_val << " ";
    }
    cout << endl;
}

void NeuralNetwork::setInputOutput(DataGroup& data) {
    for (int i = 0; i < INPUT_LAYER_NODES; ++i) {
        input_node[i]->value = data.in[i];
    }
    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) {
        output_node[i]->real_val = data.out[i];
    }
}

/*
 * using for debug to dump all of the parameters in neural network
 * */
void NeuralNetwork::printAllNode() {
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
        ss << "outputNode[" << idx++ << "] : " << node.compute_val << " ";
    }
    ss << "\n";
    std::cout << ss.str() << std::endl;
}

void NeuralNetwork::train(vector<DataGroup>& train_set, double threshold) {
    int layer, idx;
    unsigned long train_size = train_set.size();
    while(error > threshold) {
        error = 0.f;

        //1. init
        for (int i = 0; i < INPUT_LAYER_NODES; ++i) {
            input_node[i]->weight_derivative_sum.assign(HIDE_LAYER_NODES, 0.f);
        }

        for (int i = 0; i < HIDE_LAYERS; ++i) {
            unsigned long next_layer_nodes = i == HIDE_LAYERS - 1 ? OUTPUT_LAYER_NODES : HIDE_LAYER_NODES;
            for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
                hide_node[i][j]->weight_derivative_sum.assign(next_layer_nodes, 0.f);
                hide_node[i][j]->bias_derivative_sum = 0.f;
            }
        }

        for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) {
            output_node[i]->bias_derivative_sum = 0.f;
        }

        //2. train data
        for (DataGroup& data : train_set) {
            setInputOutput(data);
            forwardPropagation();
            backPropagation();
            //printAllNode();
        }


        //3. update weight and bias
        for (int i = 0; i < INPUT_LAYER_NODES; ++i) {
            for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
                input_node[i]->weight[j] -= LEARNING_RATE * input_node[i]->weight_derivative_sum[j] / train_size;
            }
        }

        for (layer = 0; layer < HIDE_LAYERS; ++layer) {
            for (idx = 0; idx < HIDE_LAYER_NODES; ++idx) {
                hide_node[layer][idx]->bias -= LEARNING_RATE * hide_node[layer][idx]->bias_derivative_sum / train_size;
                for (int next_idx = 0; next_idx < HIDE_LAYER_NODES; ++next_idx) {
                    hide_node[layer][idx]->weight[next_idx] -= LEARNING_RATE * hide_node[layer][idx]->weight_derivative_sum[next_idx] / train_size;
                }
            }
        }

        for (idx = 0; idx < OUTPUT_LAYER_NODES; ++idx) {
            output_node[idx]->bias -= LEARNING_RATE * output_node[idx]->bias_derivative_sum / train_size;
        }

        error /= train_size;
        cout << "error:" << error << endl;
    }
}
