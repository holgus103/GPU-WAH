/*
 * compress.h
 *
 *  Created on: 6 mai 2018
 *      Author: holgus103
 */
#ifndef COMPRESS_H_
#define COMPRESS_H_


template<class T>
unsigned int* compress(
		unsigned int* data_cpu,
		T dataSize,
		T* outputSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime);

#endif /* COMPRESS_H_ */
