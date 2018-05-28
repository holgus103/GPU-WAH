
#include "compress.h"
#include <stdlib.h>
#include <stdio.h>
#include "tests.h"

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
	for (int s=30;s<=30;s++) {
		for (unsigned int i=1;i<=16;i++) {
			unsigned int* dataset = new unsigned int[s*1024*1024+1];
			generateRandomData(dataset, s*1024*1024, 1<<i);
			unsigned int* compressed = compress(dataset,s*1024*1024);
		}
	}

}
