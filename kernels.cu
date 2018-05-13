#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "device_launch_parameters.h"
#include "kernels.h"
#include "tests.h"
#include <stdio.h>
#include "const.h"

#define WARP_OPERATION_DOWN(OP, NAME) \
__inline__ __device__ int NAME(int val){ \
	for (int mask = WARP_SIZE/2; mask > 0; mask /= 2)\
		val OP __shfl_xor(val, mask);\
	return val; \
}\



WARP_OPERATION_DOWN(|=, orWithinWarp);

WARP_OPERATION_DOWN(+=, sumWithinWarp);

__inline__ __device__ int localScan(int val, int id){
	for(int i = 1; i < 32; i<<=1){
		int ret = __shfl_up(val, i);
		val += id >= i ? ret : 0;
	}
	return val;
}


//__inline__ __device__ int orWithinWarp(int val) {
//  for (int mask = WARP_SIZE/2; mask > 0; mask /= 2)
//    val |= __shfl_xor(val, mask);
//  return val;
//}

//__inline__ __device__ int sumWithinWarp(int val){
//	for(int
//}

//__inline__ __device__ int countOnes(int x) {
//	x = x - ((x >> 1) & 0x55555555);
//	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
//	x = (x + (x >> 4)) & 0x0f0f0f0f;
//	x = x + (x >> 8);
//	x = x + (x >> 16);
//	return x & 0x0000003f;
//}



__global__ void compressData(unsigned int* data, unsigned int* output) {
	__shared__ int counts[32];
	// get thread id
	int id = threadIdx.x;
	int id_global = threadIdx.y *31 + id;
	unsigned int word = 0;
	// retrieve word, only first 31 threads
	if (id < WARP_SIZE - 1) {
		word = data[id_global];
	}
	// divide words into 31bit parts 
	// gets 31 - id bits from one lane above
	// and id bits from own lane
	//word = (__shfl_down(word, 1) & (ONES31 >> id)) << id | (word & TOP31ONES) >> (32 - id);
	word = ONES31 & ((__shfl_up(word, 1) >> (32 - id)) | (word << id));


	// word info variables
	int ones = 0;
	int zeros = 0;
	int literals = 0;

	// detect words with zeros and words with ones
	// is a zero fill word
	if (word == ZEROS) {
		zeros |= 1 << id;
	}

	// is a one fill word
	else if (word == ONES31) {
		ones |= 1 << id;
	}

	// exchange word information within the warp
	zeros = orWithinWarp(zeros);
	ones = orWithinWarp(ones);
	literals = ~(zeros | ones);

	// send complete information to other threads
	if (id == WARP_LEADER) {
		zeros == __shfl(zeros, 0);
		ones == __shfl(ones, 0);
		literals == __shfl(literals, 0);
	}

	int n = 0x3 << id;
	int flags = BIT31;
	bool idle = true;
	// if is not last
	if (id < 31) {
		int res = 1 << id;
		if (((n & zeros) == res || (n & ones) == res || (literals & (1 << id)) > 0)) {
			// mark endings
			flags |= 1 << id;
			idle = false;
		}
	}
	else{
		idle = false;
	}
	// exchange endings 
	flags = orWithinWarp(flags);
	int blockSize = 1;

	// calculate the number of words within a block
	if (!idle) {
		for (int i = id-1; i > 0; i--) {
			if ((flags & (1 << i)) > 0) {
				break;
			}
			blockSize++;
		}
		if (word == ONES31) {
			word = BIT3130 | blockSize;
		}
		else if (word == ZEROS) {
			word = BIT31 | blockSize;
		}
	}

	// last thread calculates the number of words and writes it to the shared array
	if(id == WARP_LEADER){
		counts[threadIdx.y] = __popc(flags);
	}

	// sync all threads within block
	__syncthreads();

	// the first warp scans the array and gets total block word size
	// then calculates offset
	if(threadIdx.y == BLOCK_LEADER){
		int count = counts[id];
		int globalOffset = localScan(count, id);
		counts[id] = globalOffset - count;
	}

	__syncthreads();
	// get global offset for warp and warp offset
	int index = counts[threadIdx.y] + __popc(((1<<id) - 1) & flags);
	output[index] = word;


}




