
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
	free(res);\
	return true;\
}

#include "compress.h"
#include "decompress.h"
#include "tests.h"
#include "const.h"
#include <stdlib.h>
#include <iostream>

/*
 * 1. 0 | 8
 * 2. 00 | 30x0
 * 3. 000 | 29x0
 * 4. 0100(4) | 28x0
 * 5. 5x0 | 27x0
 * 6. 6x1 | 26x0
 * 7. 7x1 | 25x1
 * 8. 8x0 | 24x1
 * 9..31 0...
 */
void generateTestData(unsigned int* arr, int baseIndex){
	arr[baseIndex + 0] = 8;
	arr[baseIndex + 1] = 0;
	arr[baseIndex + 3] = 4 << 28;
	arr[baseIndex + 4] = 0;
	arr[baseIndex + 5] = 63 << 26;
	arr[baseIndex + 6] = ONES;
	arr[baseIndex + 7] = ONES >> 8;
}

void generateWanderingTestData(unsigned int* arr, int baseIndex){
	arr[baseIndex] = 1;
	arr[baseIndex + 31] = 1 <<31;
	for(int i = 0; i < 30; i++){
		arr[baseIndex+ 31 + (i+1) * 32] = 1 << 30 - i;
	}
}

void generateWanderingExpectedData(unsigned int* expected, int baseIndex){
	expected[baseIndex] = 1;
	expected[baseIndex + 1] = BIT31 | 31;
	for(int i=0; i < 30; i++){
		expected[baseIndex + 2+3*i] = BIT31 | i + 1;
		expected[baseIndex + 2+3*i + 1] = 1;
		expected[baseIndex + 2+3*i + 2] = BIT31 | 30 - i;
	}
	expected[baseIndex + 91] = BIT31 | 32;
	expected[baseIndex + 92] = 1;

}

void initializeTestData(int baseIndex, unsigned int* arr){
	generateTestData(arr, baseIndex);
}

bool divideIntoWordsTest()
{
	unsigned int data[] = { 1,2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31 };
	unsigned int* results = compress(data, 31);

	unsigned int expected[32];
	expected[0] = 0x7FFFFFFF & data[0];
	for (int i = 1; i < 32; i++){
		expected[i] = 0x7FFFFFFF & ((data[i] << i) | data[i - 1] >> (32 - i));
	}

	ASSERT_32(results, expected);

	std::cout << "Division into words succeeded" << std::endl;
	free(results);
	return true;
}

bool extendDataTest() {
	unsigned int data[31] = { 0 };
	data[0] = 88;
	data[3] = 4;
	data[1] = ONES31 | BIT31;
	data[2] = ONES31 | BIT31;
	data[4] = ONES31 | BIT31;

	unsigned int expected[31] = { 0 };
	expected[0] = 88;
	expected[3] = 4;
	expected[1] = ONES31 | BIT31;
	expected[2] = ONES31 | BIT31;
	expected[4] = ONES31 | BIT31;
	for (int i = 5; i < 32; i++) {
		expected[i] = BIT31;
	}

	unsigned int* res = compress(data, 31);

	ASSERT_32(res, expected)

	std::cout << "Extension succeeded" << std::endl;
	free(res);
	return true;

}

TEST_DEC(warpCompressionTest)
	unsigned int data[31] = {0};
	generateTestData(data, 0);

//	data[0] = 8;
//	data[1] = data[2] = 0;
//	data[3] = 4 << 28;
//	data[4] = 0;
//	data[5] = 63 << 26;
//	data[6] = ONES;
//	data[7] = ONES >> 8;

	unsigned int expected[6] = {8, 3|BIT31, 4, 1|BIT31, 2|BIT3130, 24|BIT31};

	unsigned int* res = compress(data, 31);

	ASSERT(res, expected, 6);

TEST_END

TEST_DEC(blockCompressionTest)
	unsigned int data[32*31] = {0};
	for(int i = 0; i<32; i++){
		initializeTestData(i*31, data);
	}
	unsigned int * res = compress(data, 31*32);

	unsigned int help[6] = {8, 3|BIT31, 4, 1|BIT31, 2|BIT3130, 24|BIT31};
	ASSERT_MODULO(res, help, 6*32, 6);
TEST_END

TEST_DEC(blockMergeTest)
	unsigned int data[32*31] = {0};

	unsigned int expected[1] = {BIT31 | 1024};
	unsigned int* res = compress(data, 31*32);
	ASSERT(res, expected, 1);
TEST_END

TEST_DEC(blockMergeWithOnesStartsTest)
	unsigned int data[32*31] = {0};

	for(int i = 0; i < 32; i+=2){
			data[31*i] = ONES;

	}
	unsigned int* res = compress(data, 31*32);
	unsigned int help[] = {BIT3130 | 1, 1, BIT31 | 62 };
	ASSERT_MODULO(res, help, 3*16,3)
TEST_END

TEST_DEC(blockMergeAlternatingTest)
	unsigned int data[32*31] = {0};

	for(int i = 2; i < 32; i+=4){
		for(int j = 0; j < 62; j++){
			data[31*i+j] = ONES;
		}
	}

	unsigned int* res = compress(data, 31*32);
	unsigned int expected[] = {BIT31 | 64, BIT3130 | 64};
	ASSERT_MODULO(res, expected, 16, 2);
TEST_END

TEST_DEC(blockMergeFinalLiterals)
	unsigned int data[31*32] = {0};

	for(int i = 0; i < 32; i++){
		data[31*(i+1) - 1] = 88;
	}

	unsigned int* res = compress(data, 31*32);
	unsigned int expected[] = {BIT31 | 31, 44};
	ASSERT_MODULO(res, expected, 64, 2);
TEST_END

TEST_DEC(blockMergeWanderingLiterals)
	unsigned int data[31*32] = {0};

	generateWanderingTestData(data,0);

	unsigned int* res = compress(data, 31*32);
	unsigned int expected[93];

	generateWanderingExpectedData(expected, 0);

	ASSERT(res, expected, 93)

TEST_END

TEST_DEC(multiBlockTest)
	unsigned int data[2*31*32] = {0};
	generateWanderingTestData(data, 0);
	generateWanderingTestData(data, 31*32);

	unsigned int* res = compress(data, 2*31*32);
	unsigned int expected[93*2];
	generateWanderingExpectedData(expected, 0);
	generateWanderingExpectedData(expected, 93);

	ASSERT(res, expected, 186);

TEST_END

TEST_DEC(compressAndDecompressTest)
	unsigned int data[2*31*32];
	generateWanderingTestData(data, 0);
	generateWanderingTestData(data, 31*32);
	unsigned int* res = compress(data, 2*32*32);
	unsigned int* decomp = decompress(res, 186);
	free(decomp);
TEST_END



