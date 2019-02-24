/*
 * timeMeasuring.h
 *
 *  Created on: May 29, 2018
 *      Author: holgus103
 */

#ifndef TIMEMEASURING_H_
#define TIMEMEASURING_H_

#define CREATE_TIMER \
	cudaEvent_t start, stop;

#define START_TIMER \
	cudaEventCreate(&start);\
	cudaEventRecord(start,0);

#define STOP_TIMER \
	cudaEventCreate(&stop);\
	cudaEventRecord(stop,0);\
	cudaEventSynchronize(stop);

#define GET_RESULT(VAR) \
	cudaEventElapsedTime(&VAR, start,stop);


#define SAFE_ASSIGN(P_VAR, VAR) \
		if(P_VAR != NULL) (*P_VAR) = VAR;

#define FREE_ALL \
	if(data_gpu) cudaFree(data_gpu);\
	if(compressed_gpu) cudaFree(compressed_gpu);\
	if(blockCounts_gpu) cudaFree(blockCounts_gpu);\
	if(orderArray_gpu) cudaFree(orderArray_gpu);\
if(sizeCounter_gpu) cudaFree(sizeCounter_gpu);

#endif /* TIMEMEASURING_H_ */
