#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "device_launch_parameters.h"
#include "kernels.h"
#include "tests.h"
#include <stdio.h>
#include "const.h"

// some defines to avoid repetition
#define IF_LAST if(id == (warpSize - 1))

#define WARP_OPERATION_DOWN(OP, NAME) \
__inline__ __device__ int NAME(int val){ \
	for (int mask = WARP_SIZE/2; mask > 0; mask /= 2)\
		val OP __shfl_xor(val, mask);\
	return val; \
}\

WARP_OPERATION_DOWN(|=, orWithinWarp);

__inline__ __device__ int localScan(int val, int id){
	for(int i = 1; i < 32; i<<=1){
		int ret = __shfl_up(val, i);
		val += id >= i ? ret : 0;
	}
	return val;
}

__inline__ __device__ void markWordTypes(int w, int* end, int* beg, int id){
	IF_LAST{
		end[threadIdx.y] = w;
	}
	IF_LAST{
		beg[threadIdx.y] = w;
	}
}


__inline__ __device__ void writeEndingSize(int id, int* lengths, int size){
	IF_LAST{
		lengths[threadIdx.y] = size;
	}
}

__global__ void compressData(unsigned int* data, unsigned int* output) {
	// count of words for every warp
	__shared__ int counts[32];
	// lenght of the last word in a warp
	__shared__ int endLengths[32];
	// type of the last word in a warp
	__shared__ int endings[32];
	// type of the first word in a warp
	__shared__ int beginnings[32];
	// array indicating whether the last thread of a warp has been merged
	__shared__ bool merged[32] = {false};
	// shift related to warpmerging
//	__shared__ int mergeShifts[32] = {0};


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

	// mark word types for warp
	// detect words with zeros and words with ones
	// is a zero fill word
	if (word == ZEROS) {
		zeros |= 1 << id;
		markWordTypes(WORD_ZEROS, endings, beginnings, id);
	}

	// is a one fill word
	else if (word == ONES31) {
		ones |= 1 << id;
		markWordTypes(WORD_ONES, endings, beginnings, id);
	}
	else
	{
		markWordTypes(WORD_LITERAL, endings, beginnings, id);
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
//			word = BIT3130 | blockSize;
			writeEndingSize(id, endLengths, blockSize);
		}
		else if (word == ZEROS) {
//			word = BIT31 | blockSize;
			writeEndingSize(id, endLengths, blockSize);
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
	int mergeShift = 0;
	if(threadIdx.y == BLOCK_LEADER){
		int count = counts[id];
		// only execute if it's a non
		if((threadIdx.y == warpSize - 1) || (endings[id] != beginnings[id+1])){
			int i = 0;
			// calculate merge shifts
			while(true){
				// has 1 length and words match
				if(i < id && counts[id - i] == 1 && beginnings[id] == endings[id-i]){
					mergeShift++;
					merged[id - i] = true;

				}
				else if(i <= id && beginnings[id] == endings[id - i]){
					mergeShift++;
					merged[id - i] = true;
				}
				else break;
			}
			mergeShift = localScan(mergeShift, id);
			int globalOffset = localScan(count, id);
			counts[id] = globalOffset - count - mergeShift;
		}
	}

	__syncthreads();
	// get global offset for warp and warp offset
	int index = counts[threadIdx.y] + __popc(((1<<id) - 1) & flags);
	output[index] = word;


}




