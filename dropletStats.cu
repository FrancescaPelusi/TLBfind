/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "delaunayCuda.h"

int main(int argc, char **argv){
  if(argc != 7){
    fprintf(stderr, 
	    "usage: %s <dir> <dir2> <outDir> <timeStart> <timeStop> <deltaTime>\n",
	    argv[0]);
    return 1;
  }

  char *dir = argv[1];
  char *dir2 = argv[2];
  char *outDir = argv[3];
  int timeStart = atoi(argv[4]);
  int timeStop = atoi(argv[5]);
  int deltaTime = atoi(argv[6]);
  char fileNameNow[MAXFILENAME];

  delaunayType *Now = (delaunayType *)malloc(sizeof(delaunayType));
  
  for(int time = timeStart; time <= timeStop; time += deltaTime){
    int checkRead;
    do {
      checkRead = 0;
      if(time <= timeStop){
	snprintf(fileNameNow, sizeof(fileNameNow), 
		 "%s/delaunayNowTime%09d", dir, time);

	if(readCompressedFromFile(fileNameNow, Now)) checkRead += 1;
	
	if(checkRead > 0){
	  snprintf(fileNameNow, sizeof(fileNameNow), 
		   "%s/delaunayNowTime%09d", dir2, time);
	  if(readCompressedFromFile(fileNameNow, Now)) checkRead += 1;

	  else checkRead = 0;
	}
      
	if(checkRead > 0){
	  time += deltaTime;
	}
      }
    } while(checkRead != 0);    

    printBubblesToFileHostAppend(Now, time, outDir, NOW);

    delaunayType_clean(Now);
  }

  free(Now);

  return 0;
}      
