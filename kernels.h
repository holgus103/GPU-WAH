/*
 * kernels.h
 *
 *  Created on: 4 mai 2018
 *      Author: holgus103
 */

#ifndef KERNELS_H_
#define KERNELS_H_


__global__ void compressData(int* data_gpu, int* compressed_gpu, int* warpInfo);


#endif /* KERNELS_H_ */
