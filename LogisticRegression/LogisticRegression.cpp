//
// Created by ustczmk on 16/12/10.
//
#include "LogisticRegression.h"

LogisticRegression::LogisticRegression() {
    srand((unsigned)time(NULL));
    error = 100.f;
    for (int i = 0; i < INPUT_FEATURES; ++i) {
        input_node[i] = new TrainNode();
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

void LogisticRegression::forwardPropagation() {
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        double sum = output_node[i]->bias;
        for (int j = 0; j < INPUT_FEATURES; ++j) {
            sum += input_node[j]->weight[i] * input_node[j]->value;
        }
        output_node[i]->out_val = sigmoid(sum);
        double derivative_JTheta_without_regular = (1 - output_node[i]->real_val) / (1 - output_node[i]->out_val) - (output_node[i]->real_val / output_node[i]->out_val);
		derivative_JTheta_without_regular *= (1 - output_node[i]->out_val) * output_node[i]->out_val;
        output_node[i]->bias_derivative += derivative_JTheta_without_regular;
        error +=  (output_node[i]->real_val * log(output_node[i]->out_val) + (1 - output_node[i]->real_val) * log(1 - output_node[i]->out_val)) * -1;
        for (int j = 0; j < INPUT_FEATURES; ++j) {
			double derivative_JTheta = derivative_JTheta_without_regular + REGULAR_LAMBDA * input_node[j]->weight[i];
            output_node[i]->derivative_sum[j] += input_node[j]->value * derivative_JTheta;
        	error += REGULAR_LAMBDA * 0.5 * input_node[j]->weight[i] * input_node[j]->weight[i];
		}
    }
}

void LogisticRegression::predict(DataGroup &test_input) {
    setInputOutput(test_input);
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        double sum = output_node[i]->bias;
        for (int j = 0; j < INPUT_FEATURES; ++j) {
            sum += input_node[j]->weight[i] * input_node[j]->value;
        }
        test_input.out[i] = sigmoid(sum);
    }
}

void LogisticRegression::setInputOutput(DataGroup train_set) {
    for (int i = 0; i < INPUT_FEATURES; ++i) {
        input_node[i]->value = train_set.in[i];
    }
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        output_node[i]->real_val = train_set.out[i];
    }
}

void LogisticRegression::train(vector<DataGroup> train_set, double threshold) {
    while(error > threshold) {
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
        cout << "error:" << error << endl;
        cout << input_node[0]->weight[0] << " " << input_node[1]->weight[0] << endl;
    }
}
