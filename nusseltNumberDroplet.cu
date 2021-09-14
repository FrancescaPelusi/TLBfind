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
	    "usage: %s <dir> <dir2> <outDir> <timeStart> <timeStop> <timeDump> "
	    "<pbx> <pby>\n", 
	    argv[0]);
    return 1;
  }

  char *dir = argv[1];
  char *dir2 = argv[2];
  char *outDir = argv[3];
  int timeStart = atoi(argv[4]);
  int timeStop = atoi(argv[5]);
  int deltaTime = atoi(argv[6]);
  int pbcx = atoi(argv[7]);
  int pbcy = atoi(argv[8]);

  char fileNameNow[MAXFILENAME];
  char fileNameTEMP[MAXFILENAME];
  char fileNameVELO[MAXFILENAME];

  char outName[MAXFILENAME], outNameAve[MAXFILENAME];

  delaunayType *Now = (delaunayType *)malloc(sizeof(delaunayType));

  snprintf(outName, sizeof(outName),"%s/timeNusseltNumberDroplets", outDir);
  snprintf(outNameAve, sizeof(outNameAve),"%s/timeNusseltNumberDropletsAverage", outDir);  

  for(int time = timeStart; time <= timeStop; time += deltaTime){
    
    int checkRead;
    do {
      checkRead = 0;
      snprintf(fileNameNow, sizeof(fileNameNow), 
	       "%s/delaunayNowTime%09d", dir, time);
      printf("%s\n", fileNameNow);
      
      if(readCompressedFromFile(fileNameNow, Now)) checkRead += 1;
      
      if(checkRead > 0){
	checkRead = 0;
	snprintf(fileNameNow, sizeof(fileNameNow), 
		 "%s/delaunayNowTime%09d", dir2, time);
	printf("%s\n", fileNameNow);
	
	if(readCompressedFromFile(fileNameNow, Now)) checkRead += 1;
      }
      
      if(checkRead > 0){
	time += deltaTime;
      }
    } while(checkRead != 0 && time <= timeStop);    
    
    int LX = Now->h_nx;
    int LY = Now->h_ny;
    float deltaTemp = 1.0;

    scalarField2D *temperature = scalarField2D_Alloc(LX*LY);
    scalarField2D *velocityZ = scalarField2D_Alloc(LX*LY);
    scalarField2D *tempDroplets = scalarField2D_Alloc(Now->nBubbles);
    scalarField2D *veloDroplets = scalarField2D_Alloc(Now->nBubbles);
    scalarField2D *deltaTempCell = scalarField2D_Alloc(Now->nBubbles);
    scalarField2D *gradTempDroplets = scalarField2D_Alloc(Now->nBubbles);

    snprintf(fileNameTEMP, sizeof(fileNameTEMP),"temperature.%d.dat", ((time/deltaTime)-1));
    printf("%s\n", fileNameTEMP);
    
    snprintf(fileNameVELO, sizeof(fileNameVELO),"veloconf.%d.dat", ((time/deltaTime)-1));
    printf("%s\n", fileNameVELO);
    
    scalarField2D_readTemperatureFromFile(fileNameTEMP, temperature, LX, LY);
    scalarField2D_readVelocityZFromFile(fileNameVELO, velocityZ, LX, LY);
    
    scalarField2D_computeTempVeloDropSquaredCell2(Now, temperature,velocityZ,tempDroplets,veloDroplets,gradTempDroplets);
    
    scalarField2D_computeAndPrintNusseltNumberDroplets(outName, outNameAve, time, tempDroplets, veloDroplets,
						       gradTempDroplets, deltaTemp, LY);
    
    scalarField2D_Free(tempDroplets);
    scalarField2D_Free(deltaTempCell);
    scalarField2D_Free(temperature);

    scalarField2D_Free(veloDroplets);
    scalarField2D_Free(velocityZ);

    delaunayType_clean(Now);
    
  }  

  free(Now);
 
 return 0;
}
