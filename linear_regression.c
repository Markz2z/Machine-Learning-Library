/**
 * Date:03/21/2016
 * Author: Mingkun Zeng(puer)
 */

#include "stdio.h"

//calculate the average num for x
double get_avg(double x[], int len) {
	int i = 0;
	double sum = 0;
	while(i < len) {
		sum += x[i];
		++i;
	}
	return sum/len;
}

//calculate avg(x*x)
double get_avg_power_two(double x[], int len) {
	int i = 0;
	double sum = 0;
	while(i<len) {
		sum += x[i]*x[i];
		++i;
	}
	return sum/len;
}

//calculate avg(x*y)
double get_avg_arr1_arr2(double x[], double y[], int len) {
	int i = 0;
	double sum = 0;
	while(i<len) {
		sum += x[i] * y[i];
		++i;
	}
	return sum/len;
}

//simple y = ax + b model
//(x, y) represent the input data
void linear(double x[], double y[], int len) {
	double slope = (get_avg_arr1_arr2(x, y, len) - get_avg(x, len)*get_avg(y, len))/
				(get_avg_power_two(x, len) - get_avg(x, len)*get_avg(x, len));
	double intercept = get_avg(y, len) - slope * get_avg(x, len);
	printf("the best line is y = %lfx + %lf\n", slope, intercept);
}


int main() {
	double x[5] = {1, 2, 3, 4, 5};
	double y[5] = {8, 11, 14, 16, 20};
	linear(x, y, 5);
	return 1;
}