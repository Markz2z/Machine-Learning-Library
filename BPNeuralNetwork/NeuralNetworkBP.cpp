//
// Created by ustczmk on 17/2/14.
//
#include "NeuralNetworkBP.h"

NeuralNetwork::NeuralNetwork() {
    srand((unsigned)time(NULL));
    error = 100.f;

    //init input layer
    for (int i = 0; i < INPUT_LAYER_NODES; ++i) {
        input_node[i] = new InputNode();
        for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
            input_node[i]->weight.push_back(genRandom());
            input_node[i]->weight_derivative_sum.push_back(0.f);
        }
    }

    //init hide layer
    for (int i = 0; i < HIDE_LAYERS; ++i) {
        for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
            NeuronNode *neuron_node = new NeuronNode();
            int weight_cnt = i == HIDE_LAYERS - 1 ? OUTPUT_LAYER_NODES : HIDE_LAYER_NODES;
            for (int k = 0; k < weight_cnt; ++k) {
                neuron_node->weight.push_back(genRandom());
            }
            int derivative_iter = i == 0 ? INPUT_LAYER_NODES : HIDE_LAYER_NODES;
            for (int k = 0; k < derivative_iter; ++k) {
                neuron_node->weight_derivative_sum.push_back(0.f);
            }
            neuron_node->bias = genRandom();
            hide_node[i].push_back(neuron_node);
        }
    }

    //init output layer
    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) {
        output_node[i] = new OutputNode();
        output_node[i]->bias = genRandom();
    }
}

void NeuralNetwork::forwardPropagation() {
    for (int i = 0; i < HIDE_LAYERS; ++i) {
        for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
            double sum = hide_node[i][j]->bias;
            if (i==0) {
                for (int k = 0; k < INPUT_LAYER_NODES; ++k) {
                    sum += input_node[k]->value * input_node[k]->weight[j];
                }
            } else {
                for (int k = 0; k < HIDE_LAYER_NODES; ++k) {
                    sum += hide_node[i-1][k]->value * hide_node[i-1][k]->weight[j];
                }
            }
            hide_node[i][j]->value = sigmoid(sum);
        }
    }

    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) {
        double sum = output_node[i]->bias;
        for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
            sum += hide_node[HIDE_LAYERS-1][j]->value * hide_node[HIDE_LAYERS-1][j]->weight[i];
        }
        output_node[i]->compute_val = sigmoid(sum);
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

void NeuralNetwork::predict(DataGroup &test_input) {
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

void NeuralNetwork::setInputOutput(DataGroup data) {
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
    for (int i = 0; i < INPUT_LAYER_NODES; ++i) cout << "inputNode[" << i << "]:" << input_node[i]->value << " ";
    cout << endl;
    for (int i = 0; i < HIDE_LAYERS; ++i) {
        cout << "hide layer " << i << " ";
        for (int j = 0; j < HIDE_LAYER_NODES; ++j) {
            cout << " node[" << j << "]:" << hide_node[i][j]->value << " ";
        }
        cout << endl;
    }
    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) cout << "outputNode[" << i << "]:" << output_node[i]->compute_val << endl;
}

void NeuralNetwork::train(vector<DataGroup> train_set, double threshold) {
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
        for (DataGroup data : train_set) {
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