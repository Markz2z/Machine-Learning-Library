#include "stdio.h"
using namespace std;

void matrix_add(vector<vector<int>> matrix1, vector<vector<int>> matrix2[], int length, int width) {
    int i = 0, j = 0;
    while(i<length) {
        while(j<width) {
            matrix1[i][j] = matrix1[i][j] + matrix2[i][j];
            ++j;
        }
        ++i;
    }
}


//int matrix1[length][width]
//int matrix2[width][length]
//int result[length][length] = matrix1 * matrix2
vector<vector<int>> matrix_multiply(int* matrix1[], int* matrix2[], int length, int width) {
    int result[length][length];
    vector<vector<int>> result;
    int i = 0, j = 0, k = 0;
    for(i=0;i<length;++i) {
        for(j=0;j<length;++j) {
            for(k=0;k<width;++k) {
                result[i][j] = matrix1[j][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

void print(int* matrix[], int len, int wid) {
    int i = 0, j = 0;
    while(i<len) {
        while(j<wid) {
            printf("%d ", matrix[i][j]);
            ++j;
        }
        printf("\n");
        ++i;
    }
}

int main() {
    int matrix1[2][4] = {{1,2,3,4}, {2,4,6,8}};
    int matrix2[4][2] = {{1,5}, {2,6}, {3,7}, {4,8}};
    int result[2][2];
    int ** result_ptr = (int**)result;
    result_ptr = matrix_multiply(matrix1, matrix2, 2, 4);
    print(result, 2, 2);
    return 1;
}
