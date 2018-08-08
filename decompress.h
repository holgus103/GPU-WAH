/*
 * decompress.h
 *
 *  Created on: May 23, 2018
 *      Author: suchanj
 */

#ifndef DECOMPRESS_H_
#define DECOMPRESS_H_

unsigned int* decompress(
		unsigned int* data,
		unsigned long long int dataSize,
		unsigned long long int* outSize,
		unsigned long long int* offsets,
		unsigned int blocks,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime);

unsigned int* reorder(
		unsigned long long int* blockSizes,
		unsigned long long int* offsets,
		unsigned long long int blockCount,
		unsigned int* data,
		unsigned long long int dataSize,
		float* pTransferToDeviceTime,
		float* pReoderingTime,
		float* ptranserFromDeviceTime
		);



#endif /* DECOMPRESS_H_ */
