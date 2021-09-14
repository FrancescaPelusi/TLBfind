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
#include <cuda.h>
#include "delaunayCuda.h"

int main(int argc, char **argv){
  if(argc != 9){
    fprintf(stderr, 
	    "usage: %s <dir> <outDir> <timeStart> <timeStop> <deltaTime> "
	    "<dir2> <pbx> <pby>\n", 
	    argv[0]);
    return 1;
  }

  char *dir = argv[1];
  char *outDir = argv[2];
  int timeStart = atoi(argv[3]);
  int timeStop = atoi(argv[4]);
  int deltaTime = atoi(argv[5]);
  char *dir2 = argv[6];
  int pbcx = atoi(argv[7]);
  int pbcy = atoi(argv[8]);

  char fileNameNow[MAXFILENAME];
  char fileNamePast[MAXFILENAME];
  char fileNameIT[MAXFILENAME];
  char deltaName[MAXFILENAME];
  char supDeltaNormName[MAXFILENAME];
  snprintf(supDeltaNormName, sizeof(supDeltaNormName),
	   "%s/supDeltaNorm", outDir);

  delaunayType *Past = (delaunayType *)malloc(sizeof(delaunayType));
  delaunayType *Now = (delaunayType *)malloc(sizeof(delaunayType));
  int *uniqueLabels = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
  int *uniqueLabelsM1 = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
  for(int i=0; i<MAXNBUBBLE + 1; i++) uniqueLabels[i] = i;

  int *uniqueCouplesBreaking = (int *)malloc((MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*
					     sizeof(int));
  int *uniqueCouplesArising = (int *)malloc((MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*
					    sizeof(int));
  int *uniqueCouplesBreakingTime = (int *)malloc((MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*
					     sizeof(int));
  int *uniqueCouplesArisingTime = (int *)malloc((MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*
					    sizeof(int));

  memset(uniqueLabelsM1, 0, (MAXNBUBBLE + 1)*sizeof(int));
  memset(uniqueCouplesBreaking, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
  memset(uniqueCouplesArising, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
  memset(uniqueCouplesBreakingTime, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
  memset(uniqueCouplesArisingTime, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));

  for(int time = timeStart; time <= timeStop; time += deltaTime){
    int checkRead;
    do {
      checkRead = 0;
      snprintf(fileNameNow, sizeof(fileNameNow), 
	       "%s/delaunayNowTime%09d", dir, time);
      snprintf(fileNamePast, sizeof(fileNameNow), 
	       "%s/delaunayPastTime%09d", dir, time);
      snprintf(fileNameIT, sizeof(fileNameIT), 
	       "%s/delaunayIsoTriggerTime%09d", dir, time);      
      printf("%s\n", fileNameNow);
      
      if(readCompressedFromFile(fileNamePast, Past)) checkRead += 1;
      if(readCompressedFromFile(fileNameNow, Now)) checkRead += 1;
      if(readTriggerIsoFromFile(fileNameIT, Past, Now)) checkRead += 1;    
      
      if(checkRead > 0){
	checkRead = 0;
	snprintf(fileNameNow, sizeof(fileNameNow), 
		 "%s/delaunayNowTime%09d", dir2, time);
	snprintf(fileNamePast, sizeof(fileNameNow), 
		 "%s/delaunayPastTime%09d", dir2, time);
	snprintf(fileNameIT, sizeof(fileNameIT), 
		 "%s/delaunayIsoTriggerTime%09d", dir2, time);      
	printf("%s\n", fileNameNow);
	
	if(readCompressedFromFile(fileNamePast, Past)) checkRead += 1;
	if(readCompressedFromFile(fileNameNow, Now)) checkRead += 1;
	if(readTriggerIsoFromFile(fileNameIT, Past, Now)) checkRead += 1;    
      }
      
      if(checkRead > 0){
	time += deltaTime;
	memset(uniqueLabelsM1, 0, (MAXNBUBBLE + 1)*sizeof(int));
	memset(uniqueCouplesBreaking, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
	memset(uniqueCouplesArising, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
	memset(uniqueCouplesBreakingTime, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
	memset(uniqueCouplesArisingTime, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));	
	for(int i=0; i<MAXNBUBBLE + 1; i++) uniqueLabels[i] = i;
      }
    } while(checkRead != 0 && time <= timeStop);    

    for(int i=1; i<=Past->nBubbles; i++){ 
      uniqueLabels[i] = Past->isoTrigger->h_iso[uniqueLabels[i]];
    }

    memset(uniqueLabelsM1, 0, (MAXNBUBBLE + 1)*sizeof(int));
    for(int i=1; i<=Past->nBubbles; i++) uniqueLabelsM1[uniqueLabels[i]] = i;

    vectorField2D *Delta = NULL;
    if(Past->nBubbles == Now->nBubbles){
      Delta = displacementsField(Now, Past, pbcx, pbcy);
      
      snprintf(deltaName, sizeof(deltaName), "%s/DeltaField", outDir);
      displacementsFieldPrintToFile(Delta, deltaName, time,
				    uniqueLabels, Now, Past);      
      printSupNorm_vectorField2D(Delta, uniqueLabels, supDeltaNormName, time);
      vectorField2D_Free(Delta);
    }

    delaunayType_clean(Past);
    delaunayType_clean(Now);
  }

  free(Past);
  free(Now);

  return 0;
}      
