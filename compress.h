/*
 * compress.h
 *
 *  Created on: 6 mai 2018
 *      Author: holgus103
 */
#ifndef COMPRESS_H_
#define COMPRESS_H_



unsigned int* compress(
		unsigned int* data_cpu,
		unsigned long long int dataSize,
		unsigned long long int* outputSize,
		unsigned long long int** orderingArray,
		unsigned long long int* orderingLength,
		unsigned long long int** blockSizes,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime);

#endif /* COMPRESS_H_ */
