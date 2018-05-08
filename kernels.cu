#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include "kernels.h"
#include "Tests.h"
#include <stdio.h>

#define ZEROS 0
#define ONES31 0x7FFFFFFF
#define TOP31ONES 0xFFFFFFFE
#define WARP_SIZE 32
#define WARP_LEADER 0

__inline__ __device__ int reduceWithinWarp(int val) {
  for (int mask = WARP_SIZE/2; mask > 0; mask /= 2)
    val += __shfl_xor(val, mask);
  return val;
}


__global__ void compressData(int* data, int* output){
	// get thread id
	int id = blockDim.x*blockIdx.x+threadIdx.x+gridDim.x*blockDim.x*blockIdx.y;
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
	if(word == ZEROS){
		zeros |= 1 << threadIdx.x;
		word |= 1 << 31;
	}

	// is a one fill word
	else if(word == ONES31){
		ones |= 1 << threadIdx.x;
		word |= 1 << 31;
	}

	// exchange word information within the warp
	zeros = reduceWithinWarp(zeros);
	ones = reduceWithinWarp(ones);
	literals = ~(zeros | ones);
	output[id] = word;
	//// send complete information to other threads
	//if(threadIdx.x == WARP_LEADER){
	//	zeros == __shfl(zeros, 0);
	//	ones == __shfl(ones, 0);
	//	literals == __shfl(literals, 0);
	//}
	//
	//__syncthreads();
	//


	//if (word == ZEROS) {
	//	// check if it is the last word within the block
	//	
	//	// calculate proceeding zero words
	//	
	//}
	//else if (word == ONES31) {
	//	// check if it's the last word within its block
	//	// calculate proiceeding zero wordss
	//}
	//// is a tail word
	//output[id] = word;

#endif // !WORD_DIVISION_TEST

}




