#include "LogisticRegression.h"

int main() {
    LogisticRegression lr;
    vector<DataGroup> train_set;

    DataGroup train1, train2, train3, train4, test1, train5;
    //1  1 1
    train1.in.push_back(1); train1.in.push_back(1); train1.out.push_back(1);
    //1 -1 1
    train2.in.push_back(1); train2.in.push_back(-1); train2.out.push_back(1);
    //-1 1 1
    train3.in.push_back(-1); train3.in.push_back(1); train3.out.push_back(1);
    //0  0 1
    train4.in.push_back(0); train4.in.push_back(0); train4.out.push_back(1);
    //-1 -1 0
    train5.in.push_back(-1); train5.in.push_back(-1); train5.out.push_back(0);
    train_set.push_back(train1);
    train_set.push_back(train2);
    train_set.push_back(train3);
    train_set.push_back(train4);
    train_set.push_back(train5);
    lr.train(train_set, 0.0001);

    test1.in.push_back(1); test1.in.push_back(3); test1.out.push_back(-5);
    lr.predict(test1);
    cout << "result:" << test1.out[0] << endl;
    return 0;
}
