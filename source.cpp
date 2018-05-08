
#include "compress.h"
#include <stdlib.h>
#include "Tests.h"

int main()
{

#ifdef WORD_DIVISION_TEST
	testDivideIntoWordsTest();
#endif // WORD_DIVISION_TEST
	return 0;
}