/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>

int main(int argc, char **argv){

  if(argc != 7){
    fprintf(stderr,
	    "usage: %s <timeStart> <timeStop> <timeDump> <LX> <LY> <deltaTempBetweenWalls>\n",
	    argv[0]);
    return 1;
  }

  int timeStart = atoi(argv[1]);
  int timeStop = atoi(argv[2]);
  int timeDump = atoi(argv[3]);
  int LX = atoi(argv[4]);
  int LZ = atoi(argv[5]);
  float deltaTemp = atof(argv[6]);

  int x_h, z_h, x_h2, z_h2;
  float velocityX_h, velocityZ_h, temperature_h;

  float diffusivity = 1./6.;

  FILE *fileVelocity, *fileTemperature;
  FILE *fileNusseltT;
  char fileNameVELO[128], fileNameTEMP[128], fileNameOut[128];

  sprintf(fileNameOut,"timeNusseltNumber.dat");
  fileNusseltT = fopen(fileNameOut,"w");
  
  for(int time = timeStart; time <= timeStop; time+=timeDump){

    double *velocityZ = (double*)malloc((LX*LZ)*sizeof(double)); 
    double *temperature = (double*)malloc((LX*LZ)*sizeof(double)); 
    double nu1 = 0.;
    double gradT = 0.;
    
    sprintf(fileNameVELO,"veloconf.%d.dat",time/timeDump);
    fileVelocity = fopen(fileNameVELO,"r");
    
    sprintf(fileNameTEMP,"temperature.%d.dat",time/timeDump);
    fileTemperature = fopen(fileNameTEMP,"r");
    
    if(fileVelocity != NULL && fileTemperature != NULL){
      
      printf(" opening file %s \n",fileNameVELO);
      printf(" opening file %s \n",fileNameTEMP);

      for(int j = 0; j < LX; j++){
	for(int k = 0; k < LZ; k++){
	  
	  int index = k + j*LZ; 
	  
	  fscanf(fileVelocity,"%d %d %e %e", &x_h, &z_h, &velocityX_h, &velocityZ_h);
	  fscanf(fileTemperature,"%d %d %e", &x_h2, &z_h2, &temperature_h);

	  velocityZ[index] = velocityZ_h;
	  temperature[index] = temperature_h;

	}
      }
    }

    fclose(fileVelocity);
    fclose(fileTemperature);

    int counter = 0;
    for(int j = 0; j < LX; j++){
      for(int k = 1; k < (LZ-1); k++){

	int index = k + j*LZ;
	nu1 += velocityZ[index]*temperature[index];	  
	gradT += 0.5*(temperature[index+1]-temperature[index-1]);
	counter++;
      }
    }
    
    free(temperature);
    free(velocityZ);
    
    float diffusivity = 1./6.;
    double prefactor = (double)LZ/(diffusivity*deltaTemp);
    double nu1ave = nu1/(double)counter;
    double gradTave = gradT/(double)counter;

    double nu = prefactor*(nu1ave - diffusivity*gradTave);
    
    fprintf(fileNusseltT,"%d %e\n",time, nu);
    
    fflush(fileNusseltT);
  }

  fclose(fileNusseltT);
  return 0;
}
