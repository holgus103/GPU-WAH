#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include "kernels.h"
#include <stdio.h>

#define ZEROS 0
#define ONES 0x7FFFFFFF
#define WARP_SIZE 32
#define WARP_LEADER 0

__inline__ __device__
int reduceWithinWarp(int val) {
  for (int mask = WARP_SIZE/2; mask > 0; mask /= 2)
    val += __shfl_xor(val, mask);
  return val;
}

__global__ void compressData(int* data, int* output){
	// get thread id
	int id = blockDim.x*blockIdx.x+threadIdx.x+gridDim.x*blockDim.x*blockIdx.y;

	// retrieve word
	int shift = id * 31 % 32;
	int index = id * 31/32;
	int word = 0;
	word |= data[index] >> shift;
	word |= data[index+1] << (32 - shift);

	// word info variables
	int ones = 0;
	int zeros = 0;
	int literals = 0;

	// detect words with zeros and words with ones
	// is a zero fill word
	if(word == ZEROS){
		zeros |= 1 << threadIdx.x;
	}

	// is a one fill word
	else if(word == ONES){
		ones |= 1 << threadIdx.x;
	}

	// exchange word information within the warp
	zeros = reduceWithinWarp(zeros);
	ones = reduceWithinWarp(ones);
	literals = not (zeros | ones);

	// send complete information to other threads
	if(threadIdx.x == WARP_LEADER){
		zeros == __shfl(zeros, 0);
		ones == __shfl(ones, 0);
		literals == __shfl(literals, 0);
	}


	// is a tail word




}




