/*
 * kernels.h
 *
 *  Created on: 4 mai 2018
 *      Author: holgus103
 */

#ifndef KERNELS_H_
#define KERNELS_H_


__global__ void compressData(unsigned int* data_gpu, unsigned int* compressed_gpu);


#endif /* KERNELS_H_ */
