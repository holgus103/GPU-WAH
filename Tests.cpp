

#define ASSERT(RES, EX)	\
for (auto i = 0; i < 32; i++) { \
	if (RES[i] != EX[i]) { \
		std::cout << "Error at " << i << std::endl; \
		return false; \
	} \
} \

#include "compress.h"
#include "tests.h"
#include "const.h"
#include <stdlib.h>
#include <iostream>


bool divideIntoWordsTest()
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

	ASSERT(results, expected);

	std::cout << "Division into words succeeded" << std::endl;
	free(results);
	return true;
}



bool extendDataTest() {
	int data[31] = { 0 };
	data[0] = 88;
	data[3] = 4;
	data[1] = ONES31 | BIT31;
	data[2] = ONES31 | BIT31;
	data[4] = ONES31 | BIT31;

	int expected[31] = { 0 };
	expected[0] = 88;
	expected[3] = 4;
	expected[1] = ONES31 | BIT31;
	expected[2] = ONES31 | BIT31;
	expected[4] = ONES31 | BIT31;
	for (auto i = 5; i < 32; i++) {
		expected[i] = BIT31;
	}

	int* res = compress(data, 31);

	ASSERT(res, expected)

	std::cout << "Extension succeeded" << std::endl;
	free(res);
	return true;




}