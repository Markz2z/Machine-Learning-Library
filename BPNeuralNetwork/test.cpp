/*
 * Authour : Mingkun Zeng
 * Data : Jan 7th, 2018
 * */

#include "NeuralNetworkBP.h"
#include <iostream>
#include <vector>

int main() {
    std::vector<int> hidden_param = {7,7,7,7,7};
    NeuralNetwork network(2, hidden_param, 1);
    std::vector<DataGroup> train_data_vec;
    for (int i = 0; i < 100; ++i) {
        DataGroup train_data;
        train_data.in.push_back(i);
        train_data.out.push_back(10 * i);
        train_data_vec.push_back(train_data);
    }
    for (const DataGroup& data : train_data_vec) {
        std::cout << "input : " << data.in[0] << " size : " << data.in.size() << std::endl;
        std::cout << "output : " << data.out[0] << " size : " << data.out.size() << std::endl;
    }
    /*network.train(train_data_vec, 0.01);
    DataGroup test_data;
    for (int i = 1000; i < 1100; ++i) {
        test_data.in.push_back(i);
    }
    network.predict(test_data);
    for (int i = 0; i < test_data.out.size(); ++i) {
        std::cout << "In:" << test_data.in[i] << " Out:" << test_data.out[i] << std::endl;
    }*/
    return 0;
}
