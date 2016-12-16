#include <iostream>
#include "LinearRegression.h"

int main() {
    LinearRegression lr;
    vector<DataGroup> train_set;

    DataGroup train1, train2, train3, train4, train5;
    //y = 3 * x1 - 5 * x2 + 3;
    DataGroup test1;
    //1  1 1
    train1.in.push_back(1); train1.in.push_back(1); train1.out.push_back(1);
    //1 -1 1
    train2.in.push_back(2); train2.in.push_back(1); train2.out.push_back(4);
    //-1 1 1
    train3.in.push_back(-1); train3.in.push_back(3); train3.out.push_back(-15);
    //0  0 1
    train4.in.push_back(9); train4.in.push_back(1); train4.out.push_back(25);
    //-1 -1 0
    train5.in.push_back(0); train5.in.push_back(1); train5.out.push_back(-2);
    train_set.push_back(train1);
    train_set.push_back(train2);
    train_set.push_back(train3);
    train_set.push_back(train4);
    train_set.push_back(train5);
    lr.train(train_set, 1e-5);

    test1.in.push_back(1); test1.in.push_back(1); test1.out.push_back(0);
    lr.predict(test1);
    cout << "result:" << test1.out[0] << endl;
    return 0;
}
