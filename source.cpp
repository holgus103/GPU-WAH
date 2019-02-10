
#include "compress.h"
#include "decompress.h"
#include <stdlib.h>
#include <stdio.h>
#include "tests.h"
#include <fstream>
#include <iostream>

// runs tests
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
	compressAndDecompressTest<unsigned long long int>();
	compressAndDecompressTest<unsigned int>();
//	zerosTest();

//	randomDataTest<unsigned long long int>();
//	randomDataTest<unsigned int>();
	return 0;
}

// main function generating a report file
//int main(){
//
//// -- Report File initialization --
//
//  // report output file
//  std::fstream fs;
//  // open report file
//  fs.open ("results.txt", std::fstream::out | std::fstream::app);
//  // write headers
//	fs << "Original size [Int] , ";
//	fs << "Compressed size [Int] , ";
//	fs << "Decompressed size [Int] , ";
//	fs << "Density, ";
//	fs << "Compression Ratio, ";
//	fs << "Compression transfer to device [ms], ";
//	fs << "Compression time [ms], ";
//	fs << "Compression transfer from device [ms], ";
//	fs << "Decompression transfer to device [ms], ";
//	fs << "Decompression time [ms],";
//	fs << "Decompression transfer from device [ms]" << std::endl;
//
//// -- Actual processing --
//
//	// for every size - with *2 multiplication step
//	// s * 1024 is the number of blocks that will be generated and processed
//	for (int s=1;s<=256;s<<=1) {
//
//		// for every density analyzed in the previous project's code perform the tests
//		for (unsigned int i=1;i<=16;i++) {
//
//	//	--- Preparing test variables ---
//
//			// initialize all global time counters for compression
//			float c_globalTransferToDevice = 0, c_globalTransferFromDevice = 0, c_globalCompression = 0;
//			// initialize all global time counters for decompression
//			float d_globalTransferToDevice = 0, d_globalTransferFromDevice = 0 , d_globalDecompression = 0;
//			// calculate data size based on s, one block processes 31 * 32 integers
//			unsigned long long int dataSize = s*1024*31*32;
//			// data size variables
//			int globalCompressedSize, globalDecompressedSize;
//			// number of repetition for a test
//			int repetition = 10;
//
//	// 	--- Generate test data ---
//
//			// allocate memory for test data
//			unsigned int* dataset = (unsigned int*)malloc((dataSize+1) * sizeof(int));
//
//			// run data generation function
//			generateRandomData(dataset, dataSize, 1<<i);
//
//	// --- Run tests ---
//
//			// repeat the test several times
//			for(int repeat = 0; repeat < repetition; repeat++){
//
//			// ---- Initialize local test variables ----
//
//				// local counters for compression
//				float c_transferToDevice, c_transferFromDevice, c_compression;
//				// local counters for decompression
//				float d_transferToDevice, d_transferFromDevice, d_decompression;
//				// local data size - actually to be removed!
//				unsigned long long int compressedSize, decompressedSize;
//
//			// ---- Do work ----
//
//				// compress data
//				unsigned int* compressed = compress(dataset, dataSize, &compressedSize, &c_transferToDevice, &c_compression, &c_transferFromDevice);
//
//				// decompress the compressed data
//				unsigned int* decompressed = decompress(compressed, compressedSize, &decompressedSize, &d_transferToDevice, &d_decompression, &d_transferFromDevice);
//
//				// perform assertion to check if the decompressed and original data are the same
//				ASSERT(decompressed, dataset, dataSize);
//
//			// ---- Clean up after test ----
//
//				// release memory for local data
//				free(compressed);
//				free(decompressed);
//
//			// ---- Test statistics and diagnostic messages ----
//
//				std::cout << "data matches" << std::endl;
//				std::cout << " s: " << s << " i: " << i <<std::endl;
//
//				// add results to generate average values for repeated test
//				c_globalCompression += c_compression;
//				c_globalTransferFromDevice += c_transferFromDevice;
//				c_globalTransferToDevice += c_transferToDevice;
//				d_globalDecompression += d_decompression;
//				d_globalTransferFromDevice += d_transferFromDevice;
//				d_globalTransferToDevice += d_transferToDevice;
//				globalCompressedSize = compressedSize;
//				globalDecompressedSize = decompressedSize;
//			}
//
//	// --- Generate final results to the output file ---
//			fs << dataSize << ",";
//			fs << globalCompressedSize << ", ";
//			fs << globalDecompressedSize << ", ";
//			fs << i << ", ";
//			fs << ((float)globalCompressedSize)/((float)dataSize) << ", ";
//			fs << c_globalTransferToDevice / repetition << ", ";
//			fs << c_globalCompression / repetition << ", ";
//			fs << c_globalTransferFromDevice / repetition << ", ";
//			fs << d_globalTransferToDevice / repetition << ", ";
//			fs << d_globalDecompression / repetition << ", ";
//			fs << d_globalTransferFromDevice / repetition << std::endl;
//
//	// --- Global test cleanup ---
//			free(dataset);
//		}
//	}
//
//// -- Report File cleanup --
//	fs.close();
//	return 0;
//}
