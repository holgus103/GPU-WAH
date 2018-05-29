
#include "compress.h"
#include "decompress.h"
#include <stdlib.h>
#include <stdio.h>
#include "tests.h"
#include <fstream>
#include <iostream>

//int main()
//{
////	warpCompressionTest();
////	blockCompressionTest();
////	blockMergeTest();
////	blockMergeWithOnesStartsTest();
////	blockMergeAlternatingTest();
////	blockMergeFinalLiterals();
////	blockMergeWanderingLiterals();
////	multiBlockTest();
//	compressAndDecompressTest();
//	return 0;
//}



void generateRandomData(unsigned int* tab, unsigned int size, unsigned int everyN) {
	int res;
	int treshold=RAND_MAX/everyN;
	for (int i=0;i<size*32;i++) {
		int word=i>>5; // /32
		int off=i&31; // %32
		if (off==0) {
			//tab[word]=0;
			res=0;
			if (word%1000==0) printf("%f%%   \r",i*100.0/(double)(size*32-1));
			//if (word>0) printf("%x \n",tab[word-1]);
		}
		//double v=(double)rand()/(double)RAND_MAX;
		int bit=(rand()<treshold)<<off;
		res=res|bit;
		if (off==31) {
			tab[word]=res;
			//printf("%x \n",res);
		}
	}
	printf("100%%                        \n");
}

int main(){
  std::fstream fs;
  fs.open ("results.txt", std::fstream::out | std::fstream::app);

	for (int s=30;s<=30;s++) {
		for (unsigned int i=1;i<=16;i++) {
			float c_transferToDevice, c_transferFromDevice, c_compression;
			float d_transferToDevice, d_transferFromDevice, d_decompression;
			unsigned int* dataset = (unsigned int*)malloc((s*1024*1024+1) * sizeof(int));
			generateRandomData(dataset, s*1024*1024, 1<<i);
			unsigned int compressedSize, decompressedSize;
			unsigned int* compressed = compress(dataset,s*1024*1024, &compressedSize, &c_transferToDevice, &c_compression, &c_transferFromDevice);
			free(dataset);
			unsigned int* decompressed = decompress(compressed, compressedSize, &decompressedSize, &d_transferToDevice, &d_decompression, &d_transferFromDevice);
			free(compressed);
			std::cout << " s: " << s << " i: " << i <<std::endl;
			fs << "Compression:" << std::endl;
			fs << "Datasize: " <<std::endl;
			fs << "Original size: " << s*1024*1024 << std::endl;
			fs << "Compressed size: " << compressedSize << std::endl;
			fs << "Decompressed size: " << decompressedSize << std::endl;
			fs << "Compression Ratio: " << ((float)compressedSize)/((float)s*1024*1024) << std::endl;
			fs << "Compression transfer to device: " << c_transferToDevice << std::endl;
			fs << "Compression time: " << c_compression << std::endl;
			fs << "Compression transfer from device : " << c_transferFromDevice << std::endl;
			fs << "Decompression transfer to device: " << d_transferToDevice << std::endl;
			fs << "Decompression time: " << d_decompression << std::endl;
			fs << "Decompression transfer from device: " << d_transferFromDevice << std::endl;
			free(decompressed);

		}
	}
	fs.close();
	return 0;
}
