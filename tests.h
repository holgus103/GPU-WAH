#pragma once
#include <iostream>

#define ASSERT_MODULO(RES, EX, LEN, MODULO)	\
for (int i = 0; i < LEN; i++) { \
	if (RES[i] != EX[i % MODULO]) { \
		std::cout << "Error at " << i << std::endl; \
		std::cout << "Expected: " << EX[i % MODULO] << " but got: " << RES[i] <<std::endl;\
		return false; \
	} \
} \

#define ASSERT(RES, EX, LEN) ASSERT_MODULO(RES, EX, LEN, LEN)
#define ASSERT_32(RES, EX) ASSERT(RES, EX, 32)

#define TEST_DEC(NAME) \
bool NAME(){\
	std::cout << (#NAME  ": ");

#define TEST_END\
	std::cout << "succeeded" << std::endl;\
	return true;\
}


//#define WORD_DIVISION_TEST
bool divideIntoWordsTest();
bool extendDataTest();
bool warpCompressionTest();
bool blockCompressionTest();
bool blockMergeTest();
bool blockMergeWithOnesStartsTest();
bool blockMergeAlternatingTest();
bool blockMergeFinalLiterals();
bool blockMergeWanderingLiterals();
bool multiBlockTest();

template<class T>
bool compressAndDecompressTest();

template<class T>
bool randomDataTest();

bool zerosTest();
void generateRandomData(unsigned int* tab, unsigned int size, unsigned int everyN);
