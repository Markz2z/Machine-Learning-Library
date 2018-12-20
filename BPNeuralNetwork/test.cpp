/*
 * Authour : Mingkun Zeng
 * Data : Jan 7th, 2018
 * */

#include "NeuralNetworkBP.h"
#include <iostream>
#include <vector>

int main() {
    std::vector<int> hidden_param = {5, 5};
    NeuralNetwork network(2, hidden_param, 1);

    //xor [1,0] -> 1 | [0,0] -> 0 | [1,1] -> 0 | [0,1] -> 1
    network.add_train_slice({1,0}, {1});
    network.add_train_slice({0,1}, {1});
    network.add_train_slice({1,1}, {0});
    network.add_train_slice({0,0}, {0});
    network.train(0.01);

//    std::vector<double> test_input = {0.96, 0.95};
//    network.predict(test_input);
    return 0;
}
