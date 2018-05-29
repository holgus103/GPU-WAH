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

#endif /* TIMEMEASURING_H_ */
