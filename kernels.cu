#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include "kernels.h"
#include "tests.h"
#include <stdio.h>
#include "const.h"



__inline__ __device__ int reduceWithinWarp(int val) {
  for (int mask = WARP_SIZE/2; mask > 0; mask /= 2)
    val += __shfl_xor(val, mask);
  return val;
}

__inline__ __device__ int countOnes(int x) {
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0f0f0f0f;
	x = x + (x >> 8);
	x = x + (x >> 16);
	return x & 0x0000003f;
}



__global__ void compressData(int* data, int* output, int* warpInfo) {
	// get thread id
	int id = threadIdx.x;
	int word = 0;
	// retrieve word, only first 31 threads
	if (id < WARP_SIZE - 1) {
		word = data[id];
	}
	// divide words into 31bit parts 
	// gets 31 - id bits from one lane above
	// and id bits from own lane
	//word = (__shfl_down(word, 1) & (ONES31 >> id)) << id | (word & TOP31ONES) >> (32 - id);
	word = ONES31 & (__shfl_up(word, 1) >> (32 - id) | word << id);
#ifndef WORD_DIVISION_TEST


	// word info variables
	int ones = 0;
	int zeros = 0;
	int literals = 0;

	// detect words with zeros and words with ones
	// is a zero fill word
	if (word == ZEROS) {
		zeros |= 1 << threadIdx.x;
	}

	// is a one fill word
	else if (word == ONES31) {
		ones |= 1 << id;
	}

	// exchange word information within the warp
	zeros = reduceWithinWarp(zeros);
	ones = reduceWithinWarp(ones);
	literals = ~(zeros | ones);

#ifndef EXTENSION_TEST
	// send complete information to other threads
	if (id == WARP_LEADER) {
		zeros == __shfl(zeros, 0);
		ones == __shfl(ones, 0);
		literals == __shfl(literals, 0);
	}

	__syncthreads();

	int n = 0x3 << id;
	int flags = BIT31;
	bool idle = true;
	// if is not last
	if (id < 31) {
		if (((n & zeros) > 0 || (n & ones) > 0 || (literals & (1 << id)) > 0)) {
			// mark endings
			flags |= 1 << id;
			idle = false;
		}
	}
	// exchange endings 
	flags = reduceWithinWarp(flags);
	int blockSize = 1;

	// calculate the number of words within a block
	if (!idle) {
		for (auto i = id-1; i > 0; i--) {
			if ((flags & (1 << i)) > 0) {
				break;
			}
			blockSize++;
		}
	}

	if (id == WARP_LEADER) {
		warpInfo[blockIdx.x] = flags;
	}
	if (word == ONES31) {
		word = BIT31 | blockSize;
	}
	else if (word == ZEROS) {
		word = BIT3130 | blockSize;
	}
#endif // !EXTENSION_TEST



#endif // !WORD_DIVISION_TEST
	output[id] = word;
}




