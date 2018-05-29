
#include "compress.h"
#include "decompress.h"
#include <stdlib.h>
#include <stdio.h>
#include "tests.h"
#include <fstream>
#include <iostream>

int main()
{
//	warpCompressionTest();
//	blockCompressionTest();
//	blockMergeTest();
//	blockMergeWithOnesStartsTest();
//	blockMergeAlternatingTest();
//	blockMergeFinalLiterals();
//	blockMergeWanderingLiterals();
//	multiBlockTest();
//	compressAndDecompressTest();
	randomDataTest();
	return 0;
}

//int main(){
//  std::fstream fs;
//  fs.open ("results.txt", std::fstream::out | std::fstream::app);
//
//	for (int s=30;s<=30;s++) {
//		for (unsigned int i=1;i<=16;i++) {
//			float c_transferToDevice, c_transferFromDevice, c_compression;
//			float d_transferToDevice, d_transferFromDevice, d_decompression;
//			int dataSize = s*1024*31*32;
//			unsigned int* dataset = (unsigned int*)malloc((dataSize+1) * sizeof(int));
//			generateRandomData(dataset, dataSize, 1<<i);
//
//			unsigned int compressedSize, decompressedSize;
//			unsigned int* compressed = compress(dataset,dataSize, &compressedSize, &c_transferToDevice, &c_compression, &c_transferFromDevice);
//			free(dataset);
//			unsigned int* decompressed = decompress(compressed, compressedSize, &decompressedSize, &d_transferToDevice, &d_decompression, &d_transferFromDevice);
//			free(compressed);
//			std::cout << " s: " << s << " i: " << i <<std::endl;
//			fs << "Compression:" << std::endl;
//			fs << "Datasize: " <<std::endl;
//			fs << "Original size: " << dataSize << std::endl;
//			fs << "Compressed size: " << compressedSize << std::endl;
//			fs << "Decompressed size: " << decompressedSize << std::endl;
//			fs << "Compression Ratio: " << ((float)compressedSize)/((float)s*1024*1024) << std::endl;
//			fs << "Compression transfer to device: " << c_transferToDevice << std::endl;
//			fs << "Compression time: " << c_compression << std::endl;
//			fs << "Compression transfer from device : " << c_transferFromDevice << std::endl;
//			fs << "Decompression transfer to device: " << d_transferToDevice << std::endl;
//			fs << "Decompression time: " << d_decompression << std::endl;
//			fs << "Decompression transfer from device: " << d_transferFromDevice << std::endl;
//			free(decompressed);
//
//		}
//	}
//	fs.close();
//	return 0;
//}
