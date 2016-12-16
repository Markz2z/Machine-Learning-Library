//
// Created by ustczmk on 16/12/14.
//

#include "LinearRegression.h"

LinearRegression::LinearRegression() {
    srand((unsigned)time(NULL));
    error = 100.f;
    for (int i = 0; i < INPUT_FEATURES; ++i) {
        input_node[i] = new InputNode();
        for (int j = 0; j < OUTPUT_NUM; ++j) {
            input_node[i]->weight.push_back(genRandom());
        }
    }
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        output_node[i] = new OutputNode();
        output_node[i]->bias = genRandom();
        for (int j = 0; j < INPUT_FEATURES; ++j) {
            output_node[i]->derivative_sum.push_back(0.f);
        }
    }
}

void LinearRegression::forwardPropagation() {
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        double sum = output_node[i]->bias;
        for (int j = 0; j < INPUT_FEATURES; ++j) {
            sum += input_node[j]->weight[i] * input_node[j]->value;
        }
        output_node[i]->out_val = sum;
        double J_derivative_without_reg = output_node[i]->out_val - output_node[i]->real_val;
        output_node[i]->bias_derivative += J_derivative_without_reg;
        error += 0.5 * J_derivative_without_reg * J_derivative_without_reg;
        for (int j = 0; j < INPUT_FEATURES; ++j) {
			double J_derivative = J_derivative_without_reg + LAMBDA * input_node[j]->weight[i];
			output_node[i]->derivative_sum[j] += input_node[j]->value * J_derivative;
			error += input_node[j]->weight[i] * input_node[j]->weight[i] * 0.5 * LAMBDA;
		}
    }
}

void LinearRegression::predict(DataGroup &test_input) {
    setInputOutput(test_input);
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        double sum = output_node[i]->bias;
        for (int j = 0; j < INPUT_FEATURES; ++j) {
            sum += input_node[j]->weight[i] * input_node[j]->value;
        }
        test_input.out[i] = sum;
    }
}

void LinearRegression::setInputOutput(DataGroup train_set) {
    for (int i = 0; i < INPUT_FEATURES; ++i) {
        input_node[i]->value = train_set.in[i];
    }
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        output_node[i]->real_val = train_set.out[i];
    }
}

void LinearRegression::train(vector<DataGroup> train_set, double threshold) {
    int iter = 0;
	while(error > threshold) {
        ++iter;
		error = 0.f;
        for (int i = 0; i < OUTPUT_NUM; ++i) {
            output_node[i]->derivative_sum.assign(INPUT_FEATURES, 0.f);
            output_node[i]->bias_derivative = 0.f;
        }
        for (int i = 0; i < train_set.size(); ++i) {
            setInputOutput(train_set[i]);
            forwardPropagation();
        }
        //update weight
        for (int i = 0; i < INPUT_FEATURES; ++i) {
            for (int j = 0; j < OUTPUT_NUM; ++j) {
                double derivative = output_node[j]->derivative_sum[i] / train_set.size();
                input_node[i]->weight[j] -= LEARNING_RATE * derivative;
                output_node[j]->bias -= LEARNING_RATE * output_node[j]->bias_derivative / train_set.size();
            }
        }
        error /= train_set.size();
#ifdef DEBUG
        cout << "error:" << error << endl;
        for (int i = 0; i < INPUT_FEATURES; ++i) {
            for (int j = 0; j < OUTPUT_NUM; ++j) {
                cout << "weight: " <<  input_node[i]->weight[j] << " bias:" << output_node[j]->bias << " ";
            }
            cout << endl;
		}
#endif
    }
    cout << "iteration:" << iter << endl;
}
