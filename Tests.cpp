



#include "compress.h"
#include "Tests.h"
#include <stdlib.h>
#include <iostream>


bool testDivideIntoWordsTest()
{
	int data[] = { 1,2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31 };
	int* results = compress(data, 31);

	int expected[32];
	expected[0] = 0x7FFFFFFF & data[0];
	for (auto i = 1; i < 32; i++){
		expected[i] = 0x7FFFFFFF & (data[i] << i) | data[i - 1] >> (32 - i);
	}

	for (auto i = 0; i < 32; i++) {
		if (results[i] != expected[i]) {
			std::cout << "Error at " << i << std::endl;
			return false;
		}
	}
	std::cout << "Division into words succeeded" << std::endl;
	free(results);
	return true;
}