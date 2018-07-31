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
		unsigned int dataSize,
		unsigned int* outSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime);

unsigned int* reorder(
		unsigned int* blockSizes,
		unsigned int* offsets,
		int blockCount,
		unsigned int* data,
		int dataSize,
		float* pTransferToDeviceTime,
		float* pReoderingTime,
		float* ptranserFromDeviceTime
		);



#endif /* DECOMPRESS_H_ */
