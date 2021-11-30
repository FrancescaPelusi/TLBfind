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
#include <assert.h>
#include "delaunayCuda.h"
#include "timing.h"
#include <endian.h>

#define MallocCuda(p,s) cudaMalloc((p),(s))
#define malloc(s) malloc((s)); if((s)<=0) { fprintf(stderr,"Orrore in malloc: %d\n", (s)); exit(1); }

TIMER_DEF;

#define BSTH 25
#define BST 15
#define THREADS 512

__device__ __constant__ int nx, hnx, ny, hny, nla, bst;
__device__ __constant__ int d_maxVertexCount, d_linksCompressedSize;
__device__ __constant__ int d_pbcx, d_pbcy;
__device__ int trigger, chkInvert; 
__device__ __constant__ REAL threshold;

static int h_bst=BST;

enum {FALSE, TRUE};
enum {EMPTY, BUBBLE, UW, DW, LW, RW, OBS};

extern float scalarField2D_sumElements(scalarField2D *field){
  float sum = 0.;
  
  for(int i=0; i<field->N; i++){
    sum += field->scalar[i];
  }
  return sum;
}

extern float float_normalizeByIntConstant(float x, int norm){
  float res = x/(float)norm;
  return res;
}

extern float float_normalizeByFloatConstant(float x, float norm){
  float res = x/norm;
  return res;
}

extern vectorField2D *vectorField2D_minusConstant(vectorField2D *field,
				       float conX, float conY){
  vectorField2D *res = vectorField2D_Alloc(field->N);
  
  for(int i=0; i<field->N; i++){
    res->vectorX[i] = (field->vectorX[i] - conX);
    res->vectorY[i] = (field->vectorY[i] - conY);

    res->pointX[i] = field->pointX[i];
    res->pointY[i] = field->pointY[i];
  }
  return res;
}

extern scalarField2D *scalarField2D_minusConstant(scalarField2D *field, float cost){

  scalarField2D *res = scalarField2D_Alloc(field->N);
  
  for(int i=0; i<field->N; i++){
    res->scalar[i] = field->scalar[i] - cost;
    
    res->pointX[i] = field->pointX[i];
    res->pointY[i] = field->pointY[i];
  }
  return res;
}

extern scalarField2D *vectorField2D_absValue(vectorField2D *field){
  
  scalarField2D *res = scalarField2D_Alloc(field->N);
   
  for(int i=0; i<field->N; i++){
    res->scalar[i] = sqrt(field->vectorX[i]*field->vectorX[i] +
			  field->vectorY[i]*field->vectorY[i]);

    res->pointX[i] = field->pointX[i];
    res->pointY[i] = field->pointY[i];
  }
  return res;
}

extern scalarField2D *vectorField2D_fromVector2DToScalar2D(vectorField2D *field, int xy){
  
  scalarField2D *res = scalarField2D_Alloc(field->N);
  
  if(xy == 0){ 
    for(int i=0; i<field->N; i++){
      res->scalar[i] = field->vectorX[i];
      res->pointX[i] = field->pointX[i];
      res->pointY[i] = field->pointY[i];
    }
  }

  if(xy == 1){ 
    for(int i=0; i<field->N; i++){
      res->scalar[i] = field->vectorY[i];  
      res->pointX[i] = field->pointX[i];
      res->pointY[i] = field->pointY[i];
    }
  }
  return res;
}

extern int scalarField2D_countPairs(scalarField2D *field, float r, int xy){

  int index = 0; 
  
  for(int i=1; i<field->N; i++){
    for(int j=0; j<=i; j++){
       
      if(xy == 0){
	if(field->pointY[i] == field->pointY[j] && fabs(field->pointX[i] - field->pointX[j]) == r){
	  index++;
	}
      }else if(xy == 1){
	if(field->pointX[i] == field->pointX[j] && fabs(field->pointY[i] - field->pointY[j]) == r){
	  index++;
	}
	
      }
    }
  }
  return index;
}
    
extern int timeVariablePrintToFile(float variable, const char *outFile,
						  int time){
  FILE *output = fopen(outFile, "a+");
  fprintf(output, "%d %014.7e \n", time, variable);
  fclose(output);
  return 0;
}

extern void scalarField2DS_rectifyRho(REAL *h_rho, scalarField2DS *rho){
  for(int i=1; i<=rho->dimSizes[X]; i++){
    for(int j=1; j<=rho->dimSizes[Y]; j++){
      int indexH_Rho = j + i*(rho->dimSizes[Y] + 2);
      int indexRho = (i - 1) + (j - 1)*rho->dimSizes[X];
      
      rho->field->scalar[indexRho] = h_rho[indexH_Rho];
    }
  }
  return;
}

extern float scalarField2DS_findSup(scalarField2DS *sField){
  return scalarField2D_findSup(sField->field);
}

extern float scalarField2D_findSup(scalarField2D *field){
  float sup = 0.;
  for(int i=0; i<field->N; i++){
    if(i == 0) sup = field->scalar[i];
    if(field->scalar[i] > sup) sup = field->scalar[i];
  }
  return sup;
}

extern int scalarField2DS_dumpToFile(const char *fileName, scalarField2DS *sField){
  FILE *outputFile = fopen(fileName, "wb");
  fwrite(&sField->d, sizeof(int), 1, outputFile);
  fwrite(sField->dimSizes, sizeof(int), DIM, outputFile);
  fwrite(sField->localCounter, sizeof(int), sField->field->N, outputFile);
  fwrite(sField->field->scalar, sizeof(float), sField->field->N, outputFile);
  fclose(outputFile);
  return 0;
}

extern int scalarField2DS_readFromFile(const char *fileName, scalarField2DS *sField){
  FILE *inputFile = fopen(fileName, "rb");
  if (inputFile == NULL){
    fprintf(stderr, "Opening file %s error at line %d in %s\n",
	    fileName, __LINE__, __FILE__);
    return 1;
  }
  int dimSizes[DIM], d;
  fread(&d, sizeof(int), 1, inputFile);
  fread(dimSizes, sizeof(int), DIM, inputFile);

  if(sField == NULL){
    sField = scalarField2DS_Alloc(dimSizes, d);
  }
  
  fread(sField->localCounter, sizeof(int), sField->field->N, inputFile);
  fread(sField->field->scalar, sizeof(float), sField->field->N, inputFile);
  fclose(inputFile);
  return 0;
}

extern int scalarField2D_readTemperatureFromFile(char *fileName, scalarField2D *field, int LX, int LY){

  FILE *inputFile = fopen(fileName, "r");
  int x_h, y_h;
  float temperature_h;

  if (inputFile == NULL){
    fprintf(stderr, "Opening file %s error at line %d in %s\n",
	    fileName, __LINE__, __FILE__);
  }

  if(inputFile != NULL){
    
    for(int j = 0; j < LX; j++){
      for(int k = 0; k < LY; k++){  
	
	int index = k + j*LY;
	fscanf(inputFile,"%d %d %e", &x_h, &y_h, &temperature_h);
	field->pointX[index] = (float)x_h;
	field->pointY[index] = (float)y_h;
	field->scalar[index] = temperature_h;
      }
    }
  }
      
  fclose(inputFile);
  return 0;
}

extern int scalarField2D_readTemperatureProfileFromFile(char *fileName, scalarField2D *field, int LY){

  FILE *inputFile = fopen(fileName, "r");
  int y_h;
  float temperatureProfile_h;

  if (inputFile == NULL){
    fprintf(stderr, "Opening file %s error at line %d in %s\n",
	    fileName, __LINE__, __FILE__);
  }

  if(inputFile != NULL){
    
    for(int k = 1; k < (LY-1); k++){  
      
      fscanf(inputFile,"%d %e", &y_h, &temperatureProfile_h);
      field->pointY[k] = (float)y_h;
      field->scalar[k] = temperatureProfile_h;
    }
  }
  field->scalar[0] = field->scalar[1];
  field->scalar[LY-1] = field->scalar[LY-2];
      
  fclose(inputFile);
  return 0;
}

extern int scalarField2D_readVelocityZFromFile(char *fileName, scalarField2D *field, int LX, int LY){

  FILE *inputFile = fopen(fileName, "r");
  int x_h, y_h;
  float velocityX_h, velocityZ_h;

  if (inputFile == NULL){
    fprintf(stderr, "Opening file %s error at line %d in %s\n",
	    fileName, __LINE__, __FILE__);
  }

  if(inputFile != NULL){
    
    for(int j = 0; j < LX; j++){
      for(int k = 0; k < LY; k++){  
	
	int index = k + j*LY;
	fscanf(inputFile,"%d %d %e %e", &x_h, &y_h, &velocityX_h, &velocityZ_h);
	field->pointX[index] = (float)x_h;
	field->pointY[index] = (float)y_h;
	field->scalar[index] = velocityZ_h;
      }
    }
  }
      
  fclose(inputFile);
  return 0;
}

extern int *scalarField2D_computeTempVeloDropSquaredCell2(delaunayType *Now, scalarField2D *temp,scalarField2D *velo,
							  scalarField2D *tempDrop,scalarField2D *veloDrop,
							  scalarField2D *gradTempDrop){
  
  int nBubbles = Now->nBubbles;
  int LX = Now->h_nx;
  int LY = Now->h_ny;

  float x = 0., y = 0.;

  for(int i=1; i <= nBubbles; i++){

        if(Now->h_compressedXlab[i] - floor(Now->h_compressedXlab[i]) < 0.5){
	  x = floor(Now->h_compressedXlab[i]);
	}else{
	  x = ceil(Now->h_compressedXlab[i]);
	}
	Now->h_compressedXlab[i] = x;

        if(Now->h_compressedYlab[i] - floor(Now->h_compressedYlab[i]) < 0.5){
	  y = floor(Now->h_compressedYlab[i]);
	}else{
	  y = ceil(Now->h_compressedYlab[i]);
	}
	Now->h_compressedYlab[i] = y;
  }

  for(int i = 1; i <= nBubbles; i++){

    for(int j = 0; j < LX; j++){
      for(int k = 1; k < (LY-1); k++){
  	
	int index = k + j*LY;
	
	if(Now->h_compressedYlab[i] == (float)k && Now->h_compressedXlab[i] == (float)(j)){

	  tempDrop->scalar[i-1] = temp->scalar[index];
	  veloDrop->scalar[i-1] = velo->scalar[index];

	  float gradientTemp = 0.5*(temp->scalar[index+1]-temp->scalar[index-1]);
	  gradTempDrop->scalar[i-1] = gradientTemp;
      	  
	  tempDrop->pointX[i-1] = Now->h_compressedXlab[i];
	  tempDrop->pointY[i-1] = Now->h_compressedYlab[i];
	  veloDrop->pointX[i-1] = Now->h_compressedXlab[i];
	  veloDrop->pointY[i-1] = Now->h_compressedYlab[i];
	  gradTempDrop->pointX[i-1] = Now->h_compressedXlab[i];
	  gradTempDrop->pointY[i-1] = Now->h_compressedYlab[i];

	}
      }
    }
  }

  return 0;
}

extern int *scalarField2D_computeTempVeloDropSquaredCellSP(scalarField2D *temp,scalarField2D *velo,scalarField2D *tempDrop,
							   scalarField2D *veloDrop,scalarField2D *gradTempDrop,int LX, int LY,
							   int nBubblesX, int nBubblesY, float diameter, float mean,float radius){

  float x = 0., y = 0.;

  float *cmX = (float*)malloc((nBubblesX*nBubblesY)*sizeof(float));
  float *cmY = (float*)malloc((nBubblesX*nBubblesY)*sizeof(float)); 
  
  for(int j=0; j < nBubblesY; j++){
    for(int i=0; i < nBubblesX;i++){
      
      int myIndex = i + j*nBubblesX;
      
      if(j%2 == 0){
	cmX[myIndex] = mean + 0.5*diameter + i*(diameter + mean);
      }else{
	cmX[myIndex] = 0.5*mean + i*(diameter + mean);
      }
      
      cmY[myIndex] = mean + 0.5*diameter + j*0.5*sqrt(3.)*(diameter + mean);

      if(cmX[myIndex] - floor(cmX[myIndex]) < 0.5){
	x = floor(cmX[myIndex]);
      }else{
	x = ceil(cmX[myIndex]);
      }
      cmX[myIndex] = x;
      
      if(cmY[myIndex] - floor(cmY[myIndex]) < 0.5){
	y = floor(cmY[myIndex]);
      }else{
	y = ceil(cmY[myIndex]);
      }
      cmY[myIndex] = y;
    }
  }

  float *tempProfile = (float*)malloc((LY)*sizeof(float)); 

  for(int k = 0; k < LY; k++){

    float tempAve = 0.;
    int counterX = 0;

    for(int j = 0; j < LX; j++){
    
      int index = k + j*LY;
      tempAve += temp->scalar[index];
      counterX++;
    }
    tempProfile[k] = tempAve/(float)counterX;
  }

  for(int l=0; l < nBubblesY; l++){
    for(int i=0; i < nBubblesX;i++){

    int counter = 0;
    int myIndex = i + l*nBubblesX;
    
    for(int j = 0; j < LX; j++){
      for(int k = 1; k < (LY-1); k++){
  	
	int index = k + j*LY;
	
	float deltaX = fabs(cmX[myIndex]-(float)temp->pointX[index]);
	float deltaY = fabs(cmY[myIndex]-(float)temp->pointY[index]);

	if(cmY[myIndex] == (float)k){
	  float gradientTemp = 0.5*(tempProfile[k+1]-tempProfile[k-1]);
	  gradTempDrop->scalar[myIndex] = gradientTemp;
	}
	
	if(deltaX < radius && deltaY < radius){
	  
	  tempDrop->pointX[myIndex] = cmX[myIndex];
	  tempDrop->pointY[myIndex] = cmY[myIndex];
	  veloDrop->pointX[myIndex] = cmX[myIndex];
	  veloDrop->pointY[myIndex] = cmY[myIndex];
	  gradTempDrop->pointX[myIndex] = cmX[myIndex];
	  gradTempDrop->pointY[myIndex] = cmY[myIndex];

	  tempDrop->scalar[myIndex] += temp->scalar[index];
	  veloDrop->scalar[myIndex] += velo->scalar[index];

	  counter++;
	}
      }
    }
   
    tempDrop->scalar[myIndex] /= counter;
    veloDrop->scalar[myIndex] /= counter;
    }
  }

  free(tempProfile);
  free(cmX);
  free(cmY);
  
  return 0;
}

extern int scalarField2D_computeAndPrintNusseltNumberDroplets(const char *outName, const char *outNameAve, int time, scalarField2D *tempDrop,
							      scalarField2D *veloDrop,scalarField2D *gradTempDrop,
							      float deltaTemp, int LY){
  int nBubbles = tempDrop->N;
  float thermalDiff = 1./6.;
  float nusseltAve = 0.;

  FILE *out = fopen(outName, "a+");
  FILE *outTot = fopen(outNameAve, "a+");

  for(int i = 1; i <= nBubbles; i++){

    float nusselt = LY*(veloDrop->scalar[i-1]*tempDrop->scalar[i-1]-thermalDiff*gradTempDrop->scalar[i-1])/(deltaTemp*thermalDiff);
    nusseltAve += nusselt;
    
    fprintf(out, "%d %d %014.7e %014.7e %014.7e\n", 
	    time, i, tempDrop->pointX[i-1], tempDrop->pointY[i-1], nusselt);

  }
  nusseltAve /= nBubbles;
  fprintf(outTot, "%d %014.7e\n", time, nusseltAve);

  fflush(out);
  fflush(outTot);

  return 0;
}

extern void scalarField2DS_coarseGraining(scalarField2DS *sField,
					  scalarField2DS *sFieldCoarse){
  int dx, dy;
  dx = sField->dimSizes[X]/sFieldCoarse->dimSizes[X]; 
  dy = sField->dimSizes[Y]/sFieldCoarse->dimSizes[Y];
  
  for(int x=0; x<sField->dimSizes[X]; x++){
    for(int y=0; y<sField->dimSizes[Y]; y++){
      int index = x + y*sField->dimSizes[X];
      int xCoarse = x/dx, yCoarse = y/dy;
      int indexCoarse = xCoarse + yCoarse*sFieldCoarse->dimSizes[X];

      sFieldCoarse->field->scalar[indexCoarse] += sField->field->scalar[index];
      sFieldCoarse->localCounter[indexCoarse] += sField->localCounter[index];
    }
  }
			       
  return;
}

extern void scalarField2DS_dimensionalReduction(scalarField2DS *sField,
						scalarField2DS *sFieldRed){
  if(sField->dimSizes[Y] != sFieldRed->dimSizes[X]){
    fprintf(stderr, "different y dimensions, cannot Reduce data\n");
    return;
  }

  for(int y=0; y<sField->dimSizes[Y]; y++){
    for(int x=0; x<sField->dimSizes[X]; x++){
      int index2D = x + y*sField->dimSizes[X];
      sFieldRed->field->scalar[y] += sField->field->scalar[index2D];
      sFieldRed->localCounter[y]++;
    }
  }
  
  return;
}

extern int scalarField2DS_powerNfloat(scalarField2DS *A, scalarField2DS *res,
				      double power){
  if(A->dimSizes[X] != res->dimSizes[X] || A->dimSizes[Y] != res->dimSizes[Y]){
    fprintf(stderr,
	    "in function %s in file %s at line %d: ",
	    "dimensions don't match\n", __func__, __FILE__, __LINE__);
    return 1;
  }
  scalarField2D_powerNfloat(A->field, res->field, power);
  return 0;
}

extern int scalarField2D_powerNfloat(scalarField2D *A, scalarField2D *res,
				     double power){
  if(A->N != res->N){
    fprintf(stderr,
	    "in function %s in file %s at line %d: ",
	    "dimensions don't match\n", __func__, __FILE__, __LINE__);
    return 1;
  }
  
  for(int i=0; i<A->N; i++) res->scalar[i] = pow(A->scalar[i], power);
  return 0;
}

extern void scalarField2DS_setLocalCounter(scalarField2DS *sField, int value){
  for(int i=0; i<sField->field->N; i++){
      sField->localCounter[i] = value;
  }
  return;
}

extern void scalarField2DS_selfNormalize(scalarField2DS *sField){  
  for(int i=0; i<sField->field->N; i++){
    if(sField->localCounter[i] > 0){
      sField->field->scalar[i] /= (sField->localCounter[i]);
    }
  }
  return;
}


extern void scalarField2DS_normalizeByConstant(scalarField2DS *sField,
					       float norm){
  scalarField2D_normalizeByConstant(sField->field, norm);
  return;
}

extern void scalarField2D_normalizeByConstant(scalarField2D *field,
					      float norm){
  for(int i=0; i<field->N; i++) field->scalar[i] /= norm;
  return;
}

extern void scalarField2DS_printToFile(scalarField2DS *sField,
				       const char *fileName,
				       const char *mode){
  FILE *f = fopen(fileName, mode);
  scalarField2DS_printToStream(sField, f);
  fflush(f);
  fclose(f);
  return;
}

extern void scalarField2DS_printToFilePM3D(scalarField2DS *sField,
					   const char *fileName,
					   const char *mode){
  FILE *f = fopen(fileName, mode);
  scalarField2DS_printToStreamPM3D(sField, f);
  fflush(f);
  fclose(f);  
  return;
}

extern void scalarField2DS_printToStreamPM3D(scalarField2DS *sField,
					     FILE *stream){
  for(int y=0; y<sField->dimSizes[Y]; y++){
    for(int x=0; x<sField->dimSizes[X]; x++){
      int index = x + y*sField->dimSizes[X];      
      fprintf(stream, "%014.7e %014.7e %014.7e %d\n",
	      sField->field->pointX[index],
	      sField->field->pointY[index],
	      sField->field->scalar[index],
	      sField->localCounter[index]);
    }
    fprintf(stream, "\n");
  }
  return;
}

extern void scalarField2DS_printToStream(scalarField2DS *sField, FILE *stream){
  scalarField2D_printToStream(sField->field, stream);
  return;
}

extern void scalarField2D_sumOverScalarField2DS(scalarField2D *uField,
						scalarField2DS *sField){
  for(int i=0; i<uField->N; i++){
    int x = (int)(floorf(uField->pointX[i]/sField->d));
    int y = (int)(floorf(uField->pointY[i]/sField->d));
    int sIndex = x + y*sField->dimSizes[X];
    if(!isnan(uField->scalar[i])){
      sField->field->scalar[sIndex] += uField->scalar[i];
      sField->localCounter[sIndex]++;
    }
  }

  return;
}

extern int scalarField2DS_minus(scalarField2DS *A, scalarField2DS *B,
				scalarField2DS *res){
  if(A->dimSizes[X] != B->dimSizes[X] || B->dimSizes[X] != res->dimSizes[X] ||
     A->dimSizes[Y] != B->dimSizes[Y] || B->dimSizes[Y] != res->dimSizes[Y]){
    fprintf(stderr,
	    "in function %s in file %s at line %d: ",
	    "dimensions don't match\n", __func__, __FILE__, __LINE__);
    return 1;
  }
  scalarField2D_minus(A->field, B->field, res->field);
  return 0;
}

extern int scalarField2D_minus(scalarField2D *A, scalarField2D *B,
			       scalarField2D *res){
  if(A->N != B->N || B->N != res->N){
    fprintf(stderr,
	    "in function %s in file %s at line %d: ",
	    "dimensions don't match\n", __func__, __FILE__, __LINE__);
    return 1;
  }
  for(int i=0; i<A->N; i++) res->scalar[i] = (A->scalar[i] - B->scalar[i]);
  return 0;
}

extern scalarField2D *scalarField2D_diff(scalarField2D *A, 
					 scalarField2D *B){
  scalarField2D *diff = scalarField2D_Alloc(A->N);
  for(int i=0; i<A->N; i++){
    diff->scalar[i] = (A->scalar[i] - B->scalar[i]);
    diff->pointX[i] = 0.5*(A->pointX[i] + B->pointX[i]);
    diff->pointY[i] = 0.5*(A->pointY[i] + B->pointY[i]);
  }
  return diff;
}

extern int scalarField2D_printToStream(scalarField2D *field, FILE *stream){
  for(int i=0; i<field->N; i++){
    fprintf(stream, "%014.7e %014.7e %014.7e\n", 
	    field->pointX[i], field->pointY[i], field->scalar[i]);
  }
  fflush(stream);
  return 0;
}

extern int scalarField2D_printToStreamPlusTime(scalarField2D *field, FILE *stream,
					       int time){
  for(int i=0; i<field->N; i++){
    fprintf(stream, "%d %d %014.7e %014.7e %014.7e\n", 
	    time, i, field->pointX[i], field->pointY[i], field->scalar[i]);
  }
  fflush(stream);
  return 0;
}

extern int scalarField2D_printToFile(scalarField2D *field, const char *outFile, const char *mode){
  FILE *output = fopen(outFile, mode);
  scalarField2D_printToStream(field, output);
  fclose(output);
  return 0;
}

extern int scalarField2D_printToFilePlusTime(scalarField2D *field, const char *outFile, int time){
  FILE *output = fopen(outFile, "a+");
  scalarField2D_printToStreamPlusTime(field, output,time);
  fclose(output);
  return 0;
}

extern int scalarField2D_Free(scalarField2D *field){
  free(field->scalar);
  free(field->pointX);
  free(field->pointY);
  free(field);
  return 0;
}

extern scalarField2D *scalarField2D_Alloc(int N){
  scalarField2D *field = (scalarField2D *)malloc(sizeof(scalarField2D));
  if(field == NULL){
    fprintf(stderr, "Error Alloc sField in %s at line %d\n",
	    __FILE__, __LINE__);
    return NULL;
  }
  field->N = N;

  field->scalar = (float *)malloc(N*sizeof(float));
  if(field->scalar == NULL){
    fprintf(stderr, "Error Alloc sField->scalar in %s at line %d\n",
	    __FILE__, __LINE__);
    return NULL;
  }

  field->pointX = (float *)malloc(N*sizeof(float));
  if(field->pointX == NULL){
    fprintf(stderr, "Error Alloc sField->pointX in %s at line %d\n",
	    __FILE__, __LINE__);
    return NULL;
  }

  field->pointY = (float *)malloc(N*sizeof(float));
  if(field->pointY == NULL){
    fprintf(stderr, "Error Alloc sField->pointY in %s at line %d\n",
	    __FILE__, __LINE__);
    return NULL;
  }

  scalarField2D_Clean(field);

  return field;
}

extern int scalarField2D_Clean(scalarField2D *field){
  memset(field->scalar, 0, field->N*sizeof(float));
  memset(field->pointX, 0, field->N*sizeof(float));
  memset(field->pointY, 0, field->N*sizeof(float));
  return 0;
}

extern int vectorField2D_Clean(vectorField2D *field){
  memset(field->vectorX, 0, field->N*sizeof(float));
  memset(field->vectorY, 0, field->N*sizeof(float));
  memset(field->pointX, 0, field->N*sizeof(float));
  memset(field->pointY, 0, field->N*sizeof(float));

  return 0;
}

extern int scalarField2DS_Clean(scalarField2DS *field){
  memset(field->localCounter, 0, field->field->N*sizeof(float));
  scalarField2D_Clean(field->field);  
  return 0;
}

extern scalarField2DS *scalarField2DS_Alloc(int *dimSizes, int d){
  scalarField2DS *field = (scalarField2DS *)malloc(sizeof(scalarField2DS));
  if(field == NULL){
    fprintf(stderr, "Error Alloc sField in %s at line %d\n",
	    __FILE__, __LINE__);
    return NULL;
  }

  field->d = d;
  int N = 1;
  for(int i=0; i<DIM; i++){
    field->dimSizes[i] = dimSizes[i];
    N *= dimSizes[i];
  }

  field->localCounter = (int *)malloc(N*sizeof(int));
  if(field->localCounter == NULL){
    fprintf(stderr, "Error Alloc sField->localCounter in %s at line %d\n",
	    __FILE__, __LINE__);
    return NULL;
  }

  field->field = scalarField2D_Alloc(N);    

  for(int y=0; y<dimSizes[Y]; y++){
    for(int x=0; x<dimSizes[X]; x++){
      int index = x + y*dimSizes[X];
      field->field->pointX[index] = x*d;
      field->field->pointY[index] = y*d;
      field->localCounter[index] = 0;
    }
  }
  
  return field;
}

extern int scalarField2DS_Free(scalarField2DS *field){
  scalarField2D_Free(field->field);
  free(field->localCounter);
  free(field);
  return 0;
}


extern vectorField2D *vectorField2D_Alloc(int N){
  vectorField2D *field = (vectorField2D *)malloc(sizeof(vectorField2D));
  field->N = N;
  
  field->vectorX = (float *)malloc(N*sizeof(float));
  field->vectorY = (float *)malloc(N*sizeof(float));
  field->pointX = (float *)malloc(N*sizeof(float));
  field->pointY = (float *)malloc(N*sizeof(float));
  
  vectorField2D_Clean(field);
  
  return field;
}

extern int vectorField2D_Free(vectorField2D *field){
  free(field->vectorX);
  free(field->vectorY);
  free(field->pointX);
  free(field->pointY);
  free(field);
  return 0;
}

extern vectorField2D *displacementsField(delaunayType *Now, delaunayType *Past,
					 int pbcx, int pbcy){

  int *h_isoM1 = Past->isoTrigger->h_isoM1;
  int h_nx = Now->h_nx;
  int h_ny = Now->h_ny;
  int h_hnx = h_nx/2, h_hny = h_ny/2;

  vectorField2D *field = vectorField2D_Alloc(Now->nBubbles);

  for(int i=1; i<=Now->nBubbles; i++){

    float deltarX = Now->h_compressedXlab[i] - Past->h_compressedXlab[h_isoM1[i]];
    float meanX = 0.5*(Now->h_compressedXlab[i] + Past->h_compressedXlab[h_isoM1[i]]);

    if(pbcx == 1){
      meanX += (((-(fabsf(deltarX) > h_hnx)) & h_nx))*0.5;
      meanX = meanX - ((-(meanX > h_nx)) & h_nx);

      if(fabsf(deltarX) > h_hnx){
        if(deltarX > 0.) deltarX -= h_nx;
        else deltarX += h_nx;
      }
    }

    float deltarY = Now->h_compressedYlab[i] - Past->h_compressedYlab[h_isoM1[i]];
    float meanY = 0.5*(Now->h_compressedYlab[i] + Past->h_compressedYlab[h_isoM1[i]]);

    if(pbcy == 1){
      meanY += (((-(fabsf(deltarY) > h_hny)) & h_ny))*0.5;
      meanY = meanY - ((-(meanY > h_ny)) & h_ny);

      if(fabsf(deltarY) > h_hny){
        if(deltarY > 0.) deltarY -= h_ny;
        else deltarY += h_ny;
      }
    }

    field->vectorX[i - 1] = deltarX;
    field->vectorY[i - 1] = deltarY;
    field->pointX[i - 1] = meanX;
    field->pointY[i - 1] = meanY;
  }

  return field;
}

extern vectorField2D *vectorField2D_relativeDisplacements(vectorField2D *displacements,
							  delaunayType *Now, delaunayType *Past){

  int h_nx = Now->h_nx;
  int h_hnx = h_nx/2;

  int sumLinksNow = 0;
  for(int i=0; i<Now->nBubbles; i++){
    sumLinksNow += Now->h_linksCount[i + 1];
  }

  int sumLinksPastTrigger = 0;
  for(int i=0; i<Past->nBubbles; i++){
    if(Past->isoTrigger->h_localTrigger[i + 1] == 1)
      sumLinksPastTrigger += Past->h_linksCount[i + 1];
  }

  int sumLinksTot = sumLinksNow + sumLinksPastTrigger;
  int couplesNumber = 2*sumLinksTot;
  int *nowCouples = (int *)malloc(couplesNumber*sizeof(int));
  if(nowCouples == NULL){
    fprintf(stderr, "Error allocating nowCouples at line %d in file %s\n",
	    __LINE__, __FILE__);
    return NULL;
  }
  memset(nowCouples, 0, couplesNumber*sizeof(int));

  int couplesIndex = 0;

  for(int i=0; i<Now->nBubbles; i++){
    int test = TRUE;
    for(int j=0; j<Now->linksCompressedSize && test; j++){
      int indexAdj = j + (i + 1)*Now->linksCompressedSize;
      if(Now->h_compressedLinks[indexAdj] < MAXNBUBBLE){
	nowCouples[couplesIndex] = i + 1;
	nowCouples[couplesIndex + 1] = Now->h_compressedLinks[indexAdj];
	couplesIndex += 2;
      }else test = FALSE;
    }
  }

  for(int i=0; i<Past->nBubbles; i++){
    int test = TRUE;
    for(int j=0; j<Past->linksCompressedSize && test; j++){
      int indexAdj = j + (i + 1)*Past->linksCompressedSize;
      if(Past->isoTrigger->h_localTrigger[i + 1] == 1 &&
	 Past->h_compressedLinks[indexAdj] < MAXNBUBBLE){

	nowCouples[couplesIndex] = Past->isoTrigger->h_iso[i + 1];
	nowCouples[couplesIndex + 1] = Past->h_compressedLinks[indexAdj];
	couplesIndex += 2;
      }else test = FALSE;
    }
  }

  for(int i=0; i<couplesNumber; i += 2){
    for(int j=i + 2; j<couplesNumber; j += 2){
      if((nowCouples[i] == nowCouples[j] && 
	  nowCouples[i + 1] == nowCouples[j + 1]) ||
	 (nowCouples[i] == nowCouples[j + 1] && 
	  nowCouples[i + 1] == nowCouples[j])){
	nowCouples[j] = -1;
	nowCouples[j + 1] = -1;
      }
    }
  }

  int edgesNumber = 0;
  for(int i=0; i<couplesNumber; i+=2){
    if(nowCouples[i] > 0 && nowCouples[i + 1] > 0)
      edgesNumber++;
  }

  vectorField2D *relative = vectorField2D_Alloc(edgesNumber);
  int edgeIndex = 0;


  for(int i=0; i<couplesNumber; i+=2){
    if(nowCouples[i] > 0 && nowCouples[i + 1] > 0){

      float meanX = 0.5*(displacements->pointX[nowCouples[i] - 1] + 
			 displacements->pointX[nowCouples[i + 1] - 1]);

      float deltarX = (displacements->pointX[nowCouples[i] - 1] - 
		       displacements->pointX[nowCouples[i + 1] - 1]);

      meanX += (((-(fabsf(deltarX) > h_hnx)) & h_nx))*0.5;
      meanX = meanX - ((-(meanX > h_nx)) & h_nx);

      float meanY = 0.5*(displacements->pointY[nowCouples[i] - 1] + 
			 displacements->pointY[nowCouples[i + 1] - 1]);

      relative->pointX[edgeIndex] = meanX;
      relative->pointY[edgeIndex] = meanY;
      relative->vectorX[edgeIndex] = (displacements->vectorX[nowCouples[i] - 1] - 
				      displacements->vectorX[nowCouples[i + 1] - 1]);
      relative->vectorY[edgeIndex] = (displacements->vectorY[nowCouples[i] - 1] - 
				      displacements->vectorY[nowCouples[i + 1] - 1]);
      edgeIndex++;
    }
  }

  return relative;
}

extern int vectorField2D_printToFile(vectorField2D *field, const char *outFile){
  FILE *output = fopen(outFile, "w");
  vectorField2D_printToStream(field, output);
  fclose(output);
  return 0;
}

extern int vectorField2D_printToStream(vectorField2D *field, FILE *stream){
  for(int i=0; i<field->N; i++){
    fprintf(stream, "%014.7e %014.7e %014.7e %014.7e\n", 
	    field->pointX[i], field->pointY[i], 
	    field->vectorX[i], field->vectorY[i]);     
  }
  fflush(stream);
  return 0;
}

extern int displacementsFieldPrintToFile(vectorField2D *field,
					 const char *outFile, 
					 int time, int *uniqueLabels, 
					 delaunayType *Now, delaunayType *Past){
  FILE *output = fopen(outFile, "a+");
  int nBubbles = Now->nBubbles;
  int *h_isoM1 = Past->isoTrigger->h_isoM1;
  int supIndex = findSupNormIndexUnique_vectorField2D(field, uniqueLabels);

  for(int i=1; i<=nBubbles; i++){
    int index = uniqueLabels[i];

    int aveSize = 0.5*(Now->h_compressedHistogram[index] + 
		       Past->h_compressedHistogram[h_isoM1[index]]);

    int bndPast = isOnBoundaryPast(h_isoM1[index], Past->h_linksCount,
				   Past->h_compressedLinks, 
				   Past->linksCompressedSize,
				   h_isoM1);
    
    int bndNow = isOnBoundaryNow(index, Now->h_linksCount,
				 Now->h_compressedLinks, 
				 Now->linksCompressedSize);
    float norm = sqrt(field->vectorX[index - 1]*field->vectorX[index - 1] +
		      field->vectorY[index - 1]*field->vectorY[index - 1]);

    fprintf(output, "%09d %1d %1d %09d %09d %09d %014.7e %014.7e %014.7e %014.7e %014.7e %014.7e %1d\n", 
	    time, bndNow, bndPast, aveSize, i, index,
	    field->pointX[index - 1], field->pointY[index - 1], 
	    field->vectorX[index - 1], field->vectorY[index - 1],
	    norm, log(norm), (supIndex > 0 ? supIndex == index : 0));
    
  }  
  fclose(output);
  return 0;
}

extern int displacementsFieldAveragesPrintToFile(vectorField2D *field,
					 const char *outFile,int *uniqueLabels,
						 delaunayType *Now,int time){

  FILE *output = fopen(outFile, "a+");
  int nBubbles = Now->nBubbles;

  REAL displacementX = 0.;
  REAL displacementY = 0.;
  REAL displacementAbs = 0.;

  for(int i=1; i<=nBubbles; i++){

    int index = uniqueLabels[i];

    displacementX += field->vectorX[index - 1];
    displacementY += field->vectorY[index - 1];

    displacementAbs += sqrt(field->vectorX[index - 1]*field->vectorX[index - 1] +
			   field->vectorY[index - 1]*field->vectorY[index - 1]);
    
  }  

  displacementX /= nBubbles;
  displacementY /= nBubbles;
  displacementAbs /= nBubbles;

  fprintf(output, "%09d %014.7e %014.7e %014.7e\n",time,
	  displacementX, displacementY, displacementAbs);
  
  fclose(output);
  return 0;
}

extern int findSupNormIndexUnique_vectorField2D
(vectorField2D *field, int *uniqueLabels){
  int supIndex = 0;
  float supNorm = 0.;
  for(int i=1; i<field->N; i++){
    int index = uniqueLabels[i];
    float norm = sqrt(field->vectorX[index - 1]*field->vectorX[index - 1] +
		      field->vectorY[index - 1]*field->vectorY[index - 1]);
    if(norm > supNorm){
      supIndex = index;
      supNorm = norm;
    }
  }
  if(supNorm > 0.) return supIndex;
  else return 0;
}

extern int printSupNorm_vectorField2D(vectorField2D *field, int *uniqueLabels,
				      const char *fileName, int time){
  FILE *out = fopen(fileName, "a+");
  if(out == NULL){
    fprintf(stderr, "Error: cannot open file %s at line %d in file %s\n",
	    fileName, __LINE__, __FILE__);
    return 1;
  }
  int supIndex = findSupNormIndexUnique_vectorField2D(field, uniqueLabels);
  float norm = sqrt(field->vectorX[supIndex - 1]*field->vectorX[supIndex - 1] +
		    field->vectorY[supIndex - 1]*field->vectorY[supIndex - 1]);
  if(norm > 0.){
    fprintf(out, "%09d %21.15e %21.15e\n", time, norm, log(norm));
  }
  fclose(out);
  
  return 0;
}

extern void isoTriggerType_clean(isoTriggerType *isoTrigger){
  free(isoTrigger->h_iso);
  isoTrigger->h_iso = NULL;

  free(isoTrigger->h_localTrigger);
  isoTrigger->h_localTrigger = NULL;
  
  free(isoTrigger->h_isoM1);
  isoTrigger->h_isoM1 = NULL;

  return;
}

extern void delaunayType_clean(delaunayType *delaunay){

  delaunay->h_nx = delaunay->h_ny = 0;
  delaunay->nBubbles = delaunay->linksCompressedSize = 0;
  free(delaunay->h_compressedLinks);
  free(delaunay->h_compressedHistogram);
  delaunay->h_compressedLinks = delaunay->h_compressedHistogram = NULL;

  free(delaunay->h_linksCount);
  delaunay->h_linksCount = NULL;

  free(delaunay->h_compressedXlab);
  free(delaunay->h_compressedYlab);
  delaunay->h_compressedXlab = delaunay->h_compressedYlab = NULL;

  if(delaunay->isoTrigger != NULL){
    isoTriggerType_clean(delaunay->isoTrigger);
    free(delaunay->isoTrigger);
    delaunay->isoTrigger = NULL;
  }

  return;
}

extern void printBubblesToFileHost(delaunayType *delaunay, int time,
				   const char *outDir, int state){
  char outCentersOfMass[1024];
  if(state == NOW){
    snprintf(outCentersOfMass, sizeof(outCentersOfMass),
	     "%s/dropletsHostNowTime%09d", outDir, time);
  }
  if(state == PAST){
    snprintf(outCentersOfMass, sizeof(outCentersOfMass),
	     "%s/dropletsHostPastTime%09d", outDir, time);
  }

  int nBubbles = delaunay->nBubbles;
  int linksCompressedSize = delaunay->linksCompressedSize;
  int *h_compressedLinks = delaunay->h_compressedLinks;
  int *h_compressedHistogram = delaunay->h_compressedHistogram;
  int *h_linksCount = delaunay->h_linksCount;
  float *h_compressedXlab = delaunay->h_compressedXlab;
  float *h_compressedYlab = delaunay->h_compressedYlab;

  int *h_isoM1 = NULL;
  if(state == PAST){
    h_isoM1 = delaunay->isoTrigger->h_isoM1;
  }

  FILE *output = fopen(outCentersOfMass, "w");
  for(int i=1; i<=nBubbles; i++){
    int bnd = 0;
    if(state == PAST){
      bnd = isOnBoundaryPast(i, h_linksCount,
			     h_compressedLinks, 
			     linksCompressedSize,
			     h_isoM1);
    }
    if(state == NOW){
      bnd = isOnBoundaryNow(i, h_linksCount,
			    h_compressedLinks, 
			    linksCompressedSize);
    }
    fprintf(output, "%09d %09d %09d %1d %014.7e %014.7e\n",
	    i, h_compressedHistogram[i], h_linksCount[i], bnd,
	    h_compressedXlab[i], h_compressedYlab[i]);
  }
  fflush(output);
  fclose(output);

  return;
}

extern void printBubblesToFileHostAppend(delaunayType *delaunay, int time,
					 const char *outDir, int state){
  char outCentersOfMass[1024];
  if(state == NOW){
    snprintf(outCentersOfMass, sizeof(outCentersOfMass),
	     "%s/dropletsHostNow", outDir);
  }
  if(state == PAST){
    snprintf(outCentersOfMass, sizeof(outCentersOfMass),
	     "%s/dropletsHostPast", outDir);
  }

  int nBubbles = delaunay->nBubbles;
  int linksCompressedSize = delaunay->linksCompressedSize;
  int *h_compressedLinks = delaunay->h_compressedLinks;
  int *h_compressedHistogram = delaunay->h_compressedHistogram;
  int *h_linksCount = delaunay->h_linksCount;
  float *h_compressedXlab = delaunay->h_compressedXlab;
  float *h_compressedYlab = delaunay->h_compressedYlab;

  int *h_isoM1 = NULL;
  if(state == PAST){
    h_isoM1 = delaunay->isoTrigger->h_isoM1;
  }

  FILE *output = fopen(outCentersOfMass, "a+");
  for(int i=1; i<=nBubbles; i++){
    int bnd = 0;
    if(state == PAST){
      bnd = isOnBoundaryPast(i, h_linksCount,
			     h_compressedLinks, 
			     linksCompressedSize,
			     h_isoM1);
    }
    if(state == NOW){
      bnd = isOnBoundaryNow(i, h_linksCount,
			    h_compressedLinks, 
			    linksCompressedSize);
    }
    fprintf(output, "%09d %09d %09d %09d %1d %014.7e %014.7e\n",
	    time, i, h_compressedHistogram[i], h_linksCount[i], bnd,
	    h_compressedXlab[i], h_compressedYlab[i]);
  }
  
  fflush(output);
  fclose(output);

  return;
}

extern void printBubblesToFileHostAreasAveragesAppend(delaunayType *delaunay, int time,
						      const char *outDir){
  char timeAverageFile[1024];
  int nBubbles = delaunay->nBubbles;
  int *h_compressedHistogram = delaunay->h_compressedHistogram;

  snprintf(timeAverageFile, sizeof(timeAverageFile), "%s/timeAreasAverageVariance", outDir);
  FILE *output = fopen(timeAverageFile, "a+");

  REAL areasAverage = 0.;
  REAL areasAverage2 = 0.;
  REAL areasVariance = 0.;

  for(int i=1; i<=nBubbles; i++){

    areasAverage += h_compressedHistogram[i];
    areasAverage2 += (h_compressedHistogram[i]*h_compressedHistogram[i]); 

  }

  areasAverage  /= nBubbles;
  areasAverage2 /= nBubbles;

  areasVariance = (areasAverage2-(areasAverage*areasAverage))*nBubbles/(nBubbles-1);
  
  fprintf(output, "%09d %014.7e %014.7e %014.7e\n", time, areasAverage, areasVariance, areasAverage2);
  
  fflush(output);
  fclose(output);

  return;
}

extern void printBubblesToFileHostUnique(delaunayType *delaunay, int time,
					 const char *outDir, int state,
					 int *uniqueLabels, int chunk){
  char outCentersOfMass[1024];
  if(state == NOW){
    snprintf(outCentersOfMass, sizeof(outCentersOfMass),
	     "%s/dropletsHostNowChunk%09d", outDir, chunk);
  }
  if(state == PAST){
    snprintf(outCentersOfMass, sizeof(outCentersOfMass),
	     "%s/dropletsHostPastChunk%09d", outDir, chunk);
  }

  int nBubbles = delaunay->nBubbles;
  int linksCompressedSize = delaunay->linksCompressedSize;
  int *h_compressedLinks = delaunay->h_compressedLinks;
  int *h_compressedHistogram = delaunay->h_compressedHistogram;
  int *h_linksCount = delaunay->h_linksCount;
  float *h_compressedXlab = delaunay->h_compressedXlab;
  float *h_compressedYlab = delaunay->h_compressedYlab;

  int *h_isoM1 = NULL;
  if(state == PAST){
    h_isoM1 = delaunay->isoTrigger->h_isoM1;
  }

  FILE *output = fopen(outCentersOfMass, "w");
  for(int i=1; i<=nBubbles; i++){
    int bnd = 0;
    
    if(state == PAST){
      bnd = isOnBoundaryPast(i, h_linksCount,
			     h_compressedLinks, 
			     linksCompressedSize,
			     h_isoM1);
    }
    if(state == NOW){

      int index = uniqueLabels[i];
      bnd = isOnBoundaryNow(i, h_linksCount,
			    h_compressedLinks, 
			    linksCompressedSize);

      fprintf(output, "%09d %09d %09d %1d %014.7e %014.7e\n",
	      i, h_compressedHistogram[index], h_linksCount[index], bnd,
	      h_compressedXlab[index], h_compressedYlab[index]);
      
    }
  }
  fflush(output);
  fclose(output);

  return;
}

extern void printIsoTriggerToFileHost(delaunayType *Now, delaunayType *Past,
				      int time, const char *outDir){
  char outIsoTrigger[1024];

  snprintf(outIsoTrigger, sizeof(outIsoTrigger),
	   "%s/isoTriggerTime%09d", outDir, time);
  

  int nBubblesP = Past->nBubbles;
  int *h_compressedHistogramP = Past->h_compressedHistogram;
  float *h_compressedXlabP = Past->h_compressedXlab;
  float *h_compressedYlabP = Past->h_compressedYlab;
  int *h_compressedLinksP = Past->h_compressedLinks;
  int *h_linksCountP = Past->h_linksCount;
  int linksCompressedSizeP = Past->linksCompressedSize;

  int nBubblesN = Now->nBubbles;
  int *h_compressedHistogramN = Now->h_compressedHistogram;
  float *h_compressedXlabN = Now->h_compressedXlab;
  float *h_compressedYlabN = Now->h_compressedYlab;
  int *h_compressedLinksN = Now->h_compressedLinks;
  int *h_linksCountN = Now->h_linksCount;
  int linksCompressedSizeN = Now->linksCompressedSize;

  int *h_iso = Past->isoTrigger->h_iso;
  int *h_isoM1 = Past->isoTrigger->h_isoM1;
  int *h_isoCount = Past->isoTrigger->h_isoCount;

  FILE *output = fopen(outIsoTrigger, "w");
  for(int i=1; i<=nBubblesP; i++){
    int bndP = 0;
    bndP = isOnBoundaryPast(i, h_linksCountP,
			    h_compressedLinksP, 
			    linksCompressedSizeP,
			    h_isoM1);
    
    fprintf(output, 
	    "%09d %09d %1d %1d %09d %014.7e %014.7e\n",
	    i, h_iso[i], h_isoCount[h_iso[i]], 
	    bndP, h_compressedHistogramP[i], 
	    h_compressedXlabP[i], h_compressedYlabP[i]);
  }
  
  fprintf(output, "\n\n");
  
  for(int i=1; i<=nBubblesN; i++){
    int bndN = 0;
    bndN = isOnBoundaryNow(i, h_linksCountN,
			   h_compressedLinksN, 
			   linksCompressedSizeN);
    
    fprintf(output, 
	    "%09d %09d %1d %1d %09d %014.7e %014.7e\n", 
	    i, h_isoM1[i], h_isoCount[i], 
	    bndN, h_compressedHistogramN[i], 
	    h_compressedXlabN[i], h_compressedYlabN[i]);
  }

  fflush(output);
  fclose(output);

  return;
}

extern void printAdjacencyToFileHostNow(delaunayType *delaunay, int time,
					const char *outDir){
  char outDelaunayName[1024];
  
  snprintf(outDelaunayName, sizeof(outDelaunayName), 
	   "%s/newOutDelaunayHostNowTime%09d", outDir, time);

  int linksCompressedSize = delaunay->linksCompressedSize;
  int nBubbles = delaunay->nBubbles;
  int *h_compressedLinks = delaunay->h_compressedLinks;
  float *h_compressedXlab = delaunay->h_compressedXlab;
  float *h_compressedYlab = delaunay->h_compressedYlab;

  FILE *outDelaunay = fopen(outDelaunayName, "w");
  for(int i=1; i<=nBubbles; i++){
    int test = 1;
    for(int j=0; j<linksCompressedSize && test; j++){
      int index = i*linksCompressedSize + j;
      if(h_compressedLinks[index] <= nBubbles){
	fprintf(outDelaunay, "%d %014.7e %014.7e\n", 
		i, 
		h_compressedXlab[i], 
		h_compressedYlab[i]);	
	fprintf(outDelaunay, "%d %014.7e %014.7e\n",
		h_compressedLinks[index], 
		h_compressedXlab[h_compressedLinks[index]],
		h_compressedYlab[h_compressedLinks[index]]);
	fprintf(outDelaunay, "\n");
      }else test = 0;
    }
    fprintf(outDelaunay, "\n\n");
  }
  fflush(outDelaunay);
  fclose(outDelaunay);

  return;
}

extern void printRepTrianglesToFileHostNow(delaunayType *delaunay, int time,
					   const char *outDir){
  char outDelaunayName[1024];
  snprintf(outDelaunayName, sizeof(outDelaunayName), 
	   "%s/repTrianglesNowTime%09d", outDir, time);
  char outDelaunayName2[1024];
  snprintf(outDelaunayName2, sizeof(outDelaunayName2), 
	   "%s/repSplotTrianglesNowTime%09d", outDir, time);

  int linksCompressedSizeNow = delaunay->linksCompressedSize;
  int nBubbles = delaunay->nBubbles;
  int *h_compressedLinksNow = delaunay->h_compressedLinks;
  float *h_compressedXlab = delaunay->h_compressedXlab;
  float *h_compressedYlab = delaunay->h_compressedYlab;

  FILE *outDelaunay = fopen(outDelaunayName, "w");
  FILE *outDelaunay2 = fopen(outDelaunayName2, "w");
  for(int index=1; index<=nBubbles; index++){

    int testI = TRUE; 
    for(int i=0; i<linksCompressedSizeNow && testI; i++){
      int indexI = i + linksCompressedSizeNow*index;
      
      if(h_compressedLinksNow[indexI] > MAXNBUBBLE) testI = FALSE;
      else {
	int testJ = TRUE;
	for(int j=0; j<linksCompressedSizeNow && testJ; j++){
	  int indexJ = j +
	    linksCompressedSizeNow*h_compressedLinksNow[indexI];
	  
	  if(h_compressedLinksNow[indexJ] > MAXNBUBBLE) testJ = FALSE;
	  else {
	    int testK = TRUE;
	    for(int k=0; k<linksCompressedSizeNow && testK; k++){
	      int indexK = k + linksCompressedSizeNow*index;
	      if(h_compressedLinksNow[indexK] > MAXNBUBBLE) testK = FALSE;
	      else {
		if(h_compressedLinksNow[indexK] ==
		   h_compressedLinksNow[indexJ]){
		  fprintf(outDelaunay, "%d %014.7e %014.7e\n", 
			  index, 
			  h_compressedXlab[index], 
			  h_compressedYlab[index]);			
		  
		  fprintf(outDelaunay, "%d %014.7e %014.7e\n",
			  h_compressedLinksNow[indexI], 
			  h_compressedXlab[h_compressedLinksNow[indexI]],
			  h_compressedYlab[h_compressedLinksNow[indexI]]);
		  
		  fprintf(outDelaunay, "%d %014.7e %014.7e\n",
			  h_compressedLinksNow[indexK], 
			  h_compressedXlab[h_compressedLinksNow[indexK]],
			  h_compressedYlab[h_compressedLinksNow[indexK]]);
		  
		  fprintf(outDelaunay, "%d %014.7e %014.7e\n", 
			  index, 
			  h_compressedXlab[index], 
			  h_compressedYlab[index]);

		  fprintf(outDelaunay, "\n\n");

		  fprintf(outDelaunay2, "%d %014.7e %014.7e\n", 
			  index, 
			  h_compressedXlab[index], 
			  h_compressedYlab[index]);					  
		  fprintf(outDelaunay2, "%d %014.7e %014.7e\n\n",
			  h_compressedLinksNow[indexI], 
			  h_compressedXlab[h_compressedLinksNow[indexI]],
			  h_compressedYlab[h_compressedLinksNow[indexI]]);
		  
		  fprintf(outDelaunay2, "%d %014.7e %014.7e\n",
			  h_compressedLinksNow[indexK], 
			  h_compressedXlab[h_compressedLinksNow[indexK]],
			  h_compressedYlab[h_compressedLinksNow[indexK]]);
		  fprintf(outDelaunay2, "%d %014.7e %014.7e\n",
			  h_compressedLinksNow[indexK], 
			  h_compressedXlab[h_compressedLinksNow[indexK]],
			  h_compressedYlab[h_compressedLinksNow[indexK]]);

		  fprintf(outDelaunay2, "\n\n");
		  
		}
	      }
	    }
	  }
	  
	}
      }
    }
    
  }
  
  fflush(outDelaunay);
  fclose(outDelaunay);
  fflush(outDelaunay2);
  fclose(outDelaunay2);

  return;
}

extern void printAdjacencyToFileHostPast(delaunayType *delaunay, int time,
					 const char *outDir){
  char outDelaunayName[1024];
  
  snprintf(outDelaunayName, sizeof(outDelaunayName), 
	   "%s/newOutDelaunayHostPastTime%09d", outDir, time);

  int linksCompressedSize = delaunay->linksCompressedSize;
  int nBubbles = delaunay->nBubbles;
  int *h_compressedLinks = delaunay->h_compressedLinks;
  float *h_compressedXlab = delaunay->h_compressedXlab;
  float *h_compressedYlab = delaunay->h_compressedYlab;
  int *h_isoM1 = delaunay->isoTrigger->h_isoM1;

  FILE *outDelaunay = fopen(outDelaunayName, "w");
  for(int i=1; i<=nBubbles; i++){
    int test = 1;
    for(int j=0; j<linksCompressedSize && test; j++){
      int index = i*linksCompressedSize + j;
      if(h_compressedLinks[index] < nBubbles){
	fprintf(outDelaunay, "%d %014.7e %014.7e\n", 
		i, 
		h_compressedXlab[i], 
		h_compressedYlab[i]);	
	fprintf(outDelaunay, "%d %014.7e %014.7e\n",
		h_isoM1[h_compressedLinks[index]], 
		h_compressedXlab[h_isoM1[h_compressedLinks[index]]],
		h_compressedYlab[h_isoM1[h_compressedLinks[index]]]);
	fprintf(outDelaunay, "\n");
      }else test = 0;
    }
    fprintf(outDelaunay, "\n\n");
  }
  fflush(outDelaunay);
  fclose(outDelaunay);

  return;
}

extern int readTriggerIsoFromFile(char *fileName,
				  delaunayType *Past,
				  delaunayType *Now){
  FILE *input = fopen(fileName, "rb");
  if(input == NULL){
    fprintf(stderr, "error opening %s\n", fileName);
    return DE_EXIT_FAILURE;
  }

  int nBubbles = Past->nBubbles;
  int nBubblesNow = Now->nBubbles;
  Past->isoTrigger = (isoTriggerType *)malloc(sizeof(isoTriggerType));
  if(Past->isoTrigger == NULL){
    fprintf(stderr, "Error Allocating Past->isoTrigger\n");
    return DE_EXIT_FAILURE;
  }
  Past->isoTrigger->h_iso = (int *)malloc((nBubbles + 1)*sizeof(int));
  Past->isoTrigger->h_isoCount = (int *)malloc((nBubbles + 1)*sizeof(int));
  Past->isoTrigger->h_isoM1 = (int *)malloc((nBubbles + 1)*sizeof(int));
  Past->isoTrigger->h_localTrigger = (int *)malloc((nBubbles + 1)*sizeof(int));
  
  if(Past->isoTrigger->h_iso == NULL ||
     Past->isoTrigger->h_isoM1 == NULL || 
     Past->isoTrigger->h_localTrigger == NULL){
    fprintf(stderr, "No alloc readTriggerIsoFromFile\n");
    return DE_EXIT_FAILURE;
  }

  int *h_iso = Past->isoTrigger->h_iso;
  int *h_isoCount = Past->isoTrigger->h_isoCount;
  int *h_isoM1 = Past->isoTrigger->h_isoM1;
  int *h_localTrigger = Past->isoTrigger->h_localTrigger;

  fread(h_iso, sizeof(int), (nBubbles + 1), 
	input);
  fread(h_isoCount, sizeof(int), (nBubblesNow + 1), 
	input);	
  fread(h_localTrigger, sizeof(int), (nBubbles + 1), 
	input);	


  for(int i=1; i<=nBubbles; i++) h_isoM1[h_iso[i]] = i; 

  fclose(input);
  return 0;  
}

extern int readCompressedFromFile(char *fileName, 
				  delaunayType *delaunay){
  FILE *input = fopen(fileName, "rb");
  if(input == NULL){
    fprintf(stderr, "error opening %s\n", fileName);
    return DE_EXIT_FAILURE;
  }
  fread(&delaunay->h_nx, sizeof(int), 1, input);
  fread(&delaunay->h_ny, sizeof(int), 1, input);
  fread(&delaunay->nBubbles, sizeof(int), 1, input);      
  fread(&delaunay->linksCompressedSize, sizeof(int), 1, input);
  delaunay->isoTrigger = NULL;

  int nBubbles = delaunay->nBubbles;
  int linksCompressedSize = delaunay->linksCompressedSize;

  delaunay->h_compressedLinks = 
    (int *)malloc((nBubbles + 1)*linksCompressedSize*
		  sizeof(int));

  delaunay->h_compressedXlab = 
    (float *)malloc((nBubbles + 1)*sizeof(float));

  delaunay->h_compressedYlab = 
    (float *)malloc((nBubbles + 1)*sizeof(float));

  delaunay->h_compressedHistogram = 
    (int *)malloc((nBubbles + 1)*sizeof(int));

  delaunay->h_linksCount = 
    (int *)malloc((nBubbles + 1)*sizeof(int));

  if(delaunay->h_compressedLinks == NULL ||
     delaunay->h_compressedXlab == NULL || 
     delaunay->h_compressedYlab == NULL ||
     delaunay->h_compressedHistogram == NULL ||
     delaunay->h_linksCount == NULL){
    fprintf(stderr, "Error Allocating readCompressedFromFile\n");
    return DE_EXIT_FAILURE;
  }

  fread(delaunay->h_compressedLinks, sizeof(int), 
	(nBubbles + 1)*linksCompressedSize, input);

  fread(delaunay->h_compressedXlab, sizeof(float), 
	(nBubbles + 1), input);
  fread(delaunay->h_compressedYlab, sizeof(float), 
	(nBubbles + 1), input);
  fread(delaunay->h_compressedHistogram, sizeof(int), 
	(nBubbles + 1), input);
  fread(delaunay->h_linksCount, sizeof(int), (nBubbles + 1), 
	input);	


  fclose(input);
  return 0;
}

extern int isOnBoundaryPast(int index, int *h_linksCountPast,
			    int *h_compressedLinksPast, 
			    int linksCompressedSizePast,
			    int *h_isoM1){

  if(h_linksCountPast[index] < 3) return 1;

  int testI = TRUE; 
  for(int i=0; i<linksCompressedSizePast && testI; i++){
    int indexI = i + linksCompressedSizePast*index;
    
    if(h_compressedLinksPast[indexI] > MAXNBUBBLE) testI = FALSE;
    else {
      int testJ = TRUE, counter = 0;
      for(int j=0; j<linksCompressedSizePast && testJ; j++){
	int indexJ = j +
	  linksCompressedSizePast*h_isoM1[h_compressedLinksPast[indexI]];
	
	if(h_compressedLinksPast[indexJ] > MAXNBUBBLE) testJ = FALSE;
	else {
	  int testK = TRUE;
	  for(int k=0; k<linksCompressedSizePast && testK; k++){
	    int indexK = k + linksCompressedSizePast*index;
	    if(h_compressedLinksPast[indexK] > MAXNBUBBLE) testK = FALSE;
	    else {
	      counter += (h_compressedLinksPast[indexK] == 
			  h_compressedLinksPast[indexJ]);
	    }
	  }
	}
	
      }

      if(counter == 1) return 1;      
    }
  }

  return 0;
}

extern int isCoupleOnBoundary(int index0, int index1, 			      
			      int *h_compressedLinks, 
			      int linksCompressedSize){

  int testI = TRUE, counter = 0; 
  for(int i=0; i<linksCompressedSize && testI; i++){
    int indexI = i + linksCompressedSize*index0;
    
    if(h_compressedLinks[indexI] > MAXNBUBBLE) testI = FALSE;
    else {
      int testJ = TRUE;
      for(int j=0; j<linksCompressedSize && testJ; j++){
	int indexJ = j + linksCompressedSize*index1;
	
	if(h_compressedLinks[indexJ] > MAXNBUBBLE) testJ = FALSE;
	else {
	  counter += (h_compressedLinks[indexI] == 
		      h_compressedLinks[indexJ]);	    
	}		
      }
    }
  }

  if(counter == 1) return 1;      
  else return 0;
}

extern int isCoupleNow(int index0, int index1, 			      
		       int *h_compressedLinks, 
		       int linksCompressedSize){

  int testI = TRUE; 
  for(int i=0; i<linksCompressedSize && testI; i++){
    int indexI = i + linksCompressedSize*index0;
    
    if(h_compressedLinks[indexI] > MAXNBUBBLE) testI = FALSE;
    else if(index1 == h_compressedLinks[indexI]) return 1;    
  }

  if(testI == FALSE) return 0;
  return 0;
}

extern int isOnBoundaryNow(int index, int *h_linksCountNow,
			   int *h_compressedLinksNow, 
			   int linksCompressedSizeNow){

  if(h_linksCountNow[index] < 3) return 1;

  int testI = TRUE; 
  for(int i=0; i<linksCompressedSizeNow && testI; i++){
    int indexI = i + linksCompressedSizeNow*index;
    
    if(h_compressedLinksNow[indexI] > MAXNBUBBLE) testI = FALSE;
    else {
      int testJ = TRUE, counter = 0;
      for(int j=0; j<linksCompressedSizeNow && testJ; j++){
	int indexJ = j +
	  linksCompressedSizeNow*h_compressedLinksNow[indexI];
	
	if(h_compressedLinksNow[indexJ] > MAXNBUBBLE) testJ = FALSE;
	else {
	  int testK = TRUE;
	  for(int k=0; k<linksCompressedSizeNow && testK; k++){
	    int indexK = k + linksCompressedSizeNow*index;
	    if(h_compressedLinksNow[indexK] > MAXNBUBBLE) testK = FALSE;
	    else {
	      counter += (h_compressedLinksNow[indexK] == 
			  h_compressedLinksNow[indexJ]);
	    }
	  }
	}
	
      }

      if(counter == 1) return 1;      
    }
  }

  return 0;
}

extern int findAndDumpBreakingArisingDV2(int *h_localTrigger, 
					 int *h_iso, int *h_isoCount,
					 int *h_isoM1,
					 int *uniqueLabels, int *uniqueLabelsM1,
					 int *uniqueCouplesBreaking,
					 int *uniqueCouplesArising,
					 int *uniqueCouplesBreakingTime,
					 int *uniqueCouplesArisingTime,
					 int *h_compressedLinksPast,
					 float *h_compressedXlabPast,
					 float *h_compressedYlabPast,
					 int *h_compressedHistogramPast,
					 int *h_linksCountPast,
					 int *h_compressedLinksNow,
					 float *h_compressedXlabNow,
					 float *h_compressedYlabNow,
					 int *h_compressedHistogramNow,
					 int *h_linksCountNow,
					 int linksCompressedSizeNow, 
					 int linksCompressedSizePast, int time,
					 int nBubblesNow, int nBubblesPast, 
					 const char *outDir, int debug){
  static bool first = TRUE;
  static int *eventsIndices = NULL, *boundaryPast = NULL, *boundaryNow = NULL;
  static int *eventsCountNow = NULL, *eventsCountPast = NULL;
  static int *dynamicIndicesNow = NULL, *dynamicCouplesNow = NULL; 
  static int *dynamicIndicesPast = NULL, *dynamicCouplesPast = NULL; 
  static int *dynamicIndicesBoundaryNow = NULL;
  static int *dynamicIndicesBoundaryPast = NULL;
  static int *dynamicCouplesBoundaryPast = NULL;
  static int *dynamicCouplesBoundaryNow = NULL;
  static int *breakingCouples = NULL, *arisingCouples = NULL;
  static int *breakingCouplesBoundary = NULL, *arisingCouplesBoundary = NULL;
  static char breakingLinksName[MAXFILENAME];
  static char arisingLinksName[MAXFILENAME];
  static char breakingLinksBoundaryName[MAXFILENAME];
  static char arisingLinksBoundaryName[MAXFILENAME];

  static int *localNumbChangesP0 = NULL;
  static int *localNumbChangesP1 = NULL;
  static int *localNumbChangesN0 = NULL;
  static int *localNumbChangesN1 = NULL;

  if(first){
    snprintf(breakingLinksName, sizeof(breakingLinksName), 
	     "%s/breakingLinks", outDir);
    snprintf(arisingLinksName, sizeof(arisingLinksName), 
	     "%s/arisingLinks", outDir);
    snprintf(breakingLinksBoundaryName, sizeof(breakingLinksBoundaryName), 
	     "%s/breakingLinksBoundary", outDir);
    snprintf(arisingLinksBoundaryName, sizeof(arisingLinksBoundaryName), 
	     "%s/arisingLinksBoundary", outDir);

    eventsIndices = (int *)malloc(MAXNBUBBLE*sizeof(int));
    boundaryPast = (int *)malloc(MAXNBUBBLE*sizeof(int));
    boundaryNow = (int *)malloc(MAXNBUBBLE*sizeof(int));
    eventsCountNow = (int *)malloc(MAXNBUBBLE*sizeof(int));
    eventsCountPast = (int *)malloc(MAXNBUBBLE*sizeof(int));
    dynamicIndicesNow = (int *)malloc(MAXNBUBBLE*sizeof(int));
    dynamicCouplesNow = (int *)malloc(2*MAXNBUBBLE*sizeof(int));
    dynamicIndicesPast = (int *)malloc(MAXNBUBBLE*sizeof(int));
    dynamicCouplesPast = (int *)malloc(2*MAXNBUBBLE*sizeof(int));
    dynamicIndicesBoundaryNow = (int *)malloc(MAXNBUBBLE*sizeof(int));
    dynamicIndicesBoundaryPast = (int *)malloc(MAXNBUBBLE*sizeof(int));
    dynamicCouplesBoundaryNow = (int *)malloc(2*MAXNBUBBLE*sizeof(int));
    dynamicCouplesBoundaryPast = (int *)malloc(2*MAXNBUBBLE*sizeof(int));
    breakingCouples = (int *)malloc(2*MAXNBUBBLE*sizeof(int));
    arisingCouples = (int *)malloc(2*MAXNBUBBLE*sizeof(int));
    breakingCouplesBoundary = (int *)malloc(2*MAXNBUBBLE*sizeof(int));
    arisingCouplesBoundary = (int *)malloc(2*MAXNBUBBLE*sizeof(int));

    localNumbChangesP0 = (int *)malloc(MAXNBUBBLE*sizeof(int));
    localNumbChangesP1 = (int *)malloc(MAXNBUBBLE*sizeof(int));
    localNumbChangesN0 = (int *)malloc(MAXNBUBBLE*sizeof(int));
    localNumbChangesN1 = (int *)malloc(MAXNBUBBLE*sizeof(int));

    if(eventsIndices == NULL || eventsCountNow == NULL){
      fprintf(stderr, "No alloc eventsIndices\n");
      return -1;
    }
    first = FALSE;
  }

  memset(eventsIndices, 0, MAXNBUBBLE*sizeof(int));
  memset(boundaryNow, 0, MAXNBUBBLE*sizeof(int));
  memset(boundaryPast, 0, MAXNBUBBLE*sizeof(int));

  int eventsNumber = 0;
  for(int i=1; i<=nBubblesPast; i++){
    if(h_localTrigger[i] == 1){
      eventsIndices[eventsNumber] = i;
      
      boundaryPast[eventsNumber] = isOnBoundaryPast(i, h_linksCountPast,
						    h_compressedLinksPast, 
						    linksCompressedSizePast,
						    h_isoM1);

      boundaryNow[eventsNumber] = isOnBoundaryNow(h_iso[i], h_linksCountNow,
						  h_compressedLinksNow, 
						  linksCompressedSizeNow);
      eventsNumber++;
    }
  }

  if(debug == DE_DEBUG){
    int linksNumbPast = 0, linksNumbNow = 0;
    for(int i=0; i<=nBubblesNow; i++){
      linksNumbNow += h_linksCountNow[i];
    }
    for(int i=0; i<=nBubblesPast; i++){
      linksNumbPast += h_linksCountPast[i];
    }
    
    printf("numbPast: %d, numbNow: %d\n", linksNumbPast, linksNumbNow);
  }
  
  memset(eventsCountNow, 0, MAXNBUBBLE*sizeof(int));
  memset(eventsCountPast, 0, MAXNBUBBLE*sizeof(int));
  memset(dynamicIndicesNow, 0, MAXNBUBBLE*sizeof(int));
  memset(dynamicCouplesNow, 0, 2*MAXNBUBBLE*sizeof(int));
  memset(dynamicIndicesPast, 0, MAXNBUBBLE*sizeof(int));
  memset(dynamicIndicesBoundaryNow, 0, MAXNBUBBLE*sizeof(int));
  memset(dynamicIndicesBoundaryPast, 0, MAXNBUBBLE*sizeof(int));
  memset(dynamicCouplesPast, 0, 2*MAXNBUBBLE*sizeof(int));
  memset(dynamicCouplesBoundaryNow, 0, 2*MAXNBUBBLE*sizeof(int));
  memset(dynamicCouplesBoundaryPast, 0, 2*MAXNBUBBLE*sizeof(int));
  memset(breakingCouples, 0, 2*MAXNBUBBLE*sizeof(int));
  memset(arisingCouples, 0, 2*MAXNBUBBLE*sizeof(int));
  memset(breakingCouplesBoundary, 0, 2*MAXNBUBBLE*sizeof(int));
  memset(arisingCouplesBoundary, 0, 2*MAXNBUBBLE*sizeof(int));

  int dynamicIndicesNumberPast = 0;
  int dynamicIndicesNumberNow = 0;
  for(int i=0; i<eventsNumber; i++){
    if(debug == DE_DEBUG)
      printf("%04d: ", h_iso[eventsIndices[i]]);
    for(int j=0; j<eventsNumber; j++){
      int test = TRUE;
      if(i == j) continue;
      for(int k=0; k<linksCompressedSizePast && test; k++){
	int indexPast = k + linksCompressedSizePast*eventsIndices[j];
	if(debug == DE_DEBUG)
	  printf("{%010d} ", 
		 h_compressedLinksPast[indexPast]);
	if(h_compressedLinksPast[indexPast] == h_iso[eventsIndices[i]]){
	  eventsCountPast[i]++; 
	  if(debug == DE_DEBUG) printf("[yesPast!] ");
	}
	if(h_compressedLinksPast[indexPast] > MAXNBUBBLE) test = FALSE;
      }      
      if(debug == DE_DEBUG) printf("\n");
      test = TRUE;
      for(int k=0; k<linksCompressedSizeNow && test; k++){
	int indexNow = k + linksCompressedSizeNow*h_iso[eventsIndices[j]];
	if(debug == DE_DEBUG) 
	  printf("{%010d} ", 
		 h_compressedLinksNow[indexNow]);
	if(h_compressedLinksNow[indexNow] == h_iso[eventsIndices[i]]){
	  eventsCountNow[i]++; 
	  if(debug == DE_DEBUG) printf("[yesNow!] ");
	}
	if(h_compressedLinksNow[indexNow] > MAXNBUBBLE) test = FALSE;
      }      
      if(debug == DE_DEBUG) printf("\n\n");
    }

    if(eventsCountNow[i] >= 3 || 
       (eventsCountNow[i] > 0 && boundaryNow[i] == 1)){
      dynamicIndicesNow[dynamicIndicesNumberNow] = eventsIndices[i];
      dynamicIndicesBoundaryNow[dynamicIndicesNumberNow] = boundaryNow[i];
      dynamicIndicesNumberNow++;
    }
    if(eventsCountPast[i] >= 3 || 
       (eventsCountPast[i] > 0 && boundaryPast[i] == 1)){
      dynamicIndicesPast[dynamicIndicesNumberPast] = eventsIndices[i];
      dynamicIndicesBoundaryPast[dynamicIndicesNumberPast] = boundaryPast[i];
      dynamicIndicesNumberPast++;
    }
  }

  if(debug == DE_DEBUG){
    printf("eventsCountP: ");
    for(int i=0; i<eventsNumber; i++){
      printf("%d: %d, ", h_iso[eventsIndices[i]], eventsCountPast[i]);
    }
    printf("\n");
    printf("eventsCountN: ");
    for(int i=0; i<eventsNumber; i++){
      printf("%d: %d, ", eventsIndices[i], eventsCountNow[i]);
    }
    printf("\n");
    
    printf("dynamicIndicesP: ");
    for(int i=0; i<dynamicIndicesNumberPast; i++){
      printf("%d ", h_iso[dynamicIndicesPast[i]]);
    }
    printf("\n");
    printf("dynamicIndicesN: ");
    for(int i=0; i<dynamicIndicesNumberNow; i++){
    printf("%d ", dynamicIndicesNow[i]);
    }
    printf("\n");
  }

  int dynamicCouplesNumberPast = 0;
  for(int i=0; i<dynamicIndicesNumberPast; i++){
    for(int j=0; j<dynamicIndicesNumberPast; j++){
      int test = TRUE;
      if(i == j) continue;
      if(debug == DE_DEBUG) printf("%d %d\n", i, j);
      for(int k=0; k<linksCompressedSizePast && test; k++){
	int index = k + linksCompressedSizePast*dynamicIndicesPast[j];
	if(h_compressedLinksPast[index] == h_iso[dynamicIndicesPast[i]]){
	  dynamicCouplesPast[dynamicCouplesNumberPast] = dynamicIndicesPast[i];
	  dynamicCouplesPast[dynamicCouplesNumberPast + 1] = 
	    dynamicIndicesPast[j];

	  dynamicCouplesBoundaryPast[dynamicCouplesNumberPast] = 
	    dynamicIndicesBoundaryPast[i];

	  dynamicCouplesBoundaryPast[dynamicCouplesNumberPast + 1] = 
	    dynamicIndicesBoundaryPast[j];

	  dynamicCouplesNumberPast += 2;
	  if(debug == DE_DEBUG) printf("%d %d yes\n", j, k);
	}
	if(h_compressedLinksPast[index] > MAXNBUBBLE) test = FALSE;
      }
    }
    if(debug == DE_DEBUG) printf("i: %d\n\n", i);    
  }

  int dynamicCouplesNumberNow = 0;
  for(int i=0; i<dynamicIndicesNumberNow; i++){
    for(int j=0; j<dynamicIndicesNumberNow; j++){
      int test = TRUE;
      if(debug == DE_DEBUG) printf("%d %d\n", i, j);
      for(int k=0; k<linksCompressedSizeNow && test; k++){
	int index = k + linksCompressedSizeNow*h_iso[dynamicIndicesNow[j]];
	if(h_compressedLinksNow[index] == h_iso[dynamicIndicesNow[i]]){
	  dynamicCouplesNow[dynamicCouplesNumberNow] = dynamicIndicesNow[i];
	  dynamicCouplesNow[dynamicCouplesNumberNow + 1] = 
	    dynamicIndicesNow[j];

	  dynamicCouplesBoundaryNow[dynamicCouplesNumberNow] = 
	    dynamicIndicesBoundaryNow[i];

	  dynamicCouplesBoundaryNow[dynamicCouplesNumberNow + 1] = 
	    dynamicIndicesBoundaryNow[j];

	  dynamicCouplesNumberNow += 2;
	  if(debug == DE_DEBUG) printf("%d %d yes\n", j, k);
	}
	if(h_compressedLinksNow[index] > MAXNBUBBLE) test = FALSE;
      }
    }
    if(debug == DE_DEBUG) printf("i: %d\n\n", i);    
  }

  for(int i=0; i<dynamicCouplesNumberPast; i+=2){
    for(int j=i + 2; j<dynamicCouplesNumberPast; j+=2){
      if((dynamicCouplesPast[i] == dynamicCouplesPast[j] &&
	  dynamicCouplesPast[i + 1] == dynamicCouplesPast[j + 1]) ||
	 (dynamicCouplesPast[i] == dynamicCouplesPast[j + 1] &&
	  dynamicCouplesPast[i + 1] == dynamicCouplesPast[j])){

	dynamicCouplesPast[j] = -1;
	dynamicCouplesPast[j + 1] = -1;
      }
    }
  }

  for(int i=0; i<dynamicCouplesNumberNow; i+=2){
    for(int j=i + 2; j<dynamicCouplesNumberNow; j+=2){
      if((dynamicCouplesNow[i] == dynamicCouplesNow[j] &&
	  dynamicCouplesNow[i + 1] == dynamicCouplesNow[j + 1]) ||
	 (dynamicCouplesNow[i] == dynamicCouplesNow[j + 1] &&
	  dynamicCouplesNow[i + 1] == dynamicCouplesNow[j])){

	dynamicCouplesNow[j] = -1;
	dynamicCouplesNow[j + 1] = -1;
      }
    }
  }

  if(debug == DE_DEBUG){
    printf("dynamicCouplesPast: ");
    for(int i=0; i<dynamicCouplesNumberPast; i+=2){
      if(dynamicCouplesPast[i] > 0){
	printf("{%04d [%d], %04d [%d]} ", 
	       h_iso[dynamicCouplesPast[i]], dynamicCouplesBoundaryPast[i],
	       h_iso[dynamicCouplesPast[i + 1]], dynamicCouplesBoundaryPast[i]);
      }
    }
    printf("\n");
    
    printf("dynamicCouplesNow: ");
    for(int i=0; i<dynamicCouplesNumberNow; i+=2){
      if(dynamicCouplesNow[i] > 0){
	printf("{%04d [%d], %04d [%d]} ", 
	       dynamicCouplesNow[i], dynamicCouplesBoundaryNow[i],
	       dynamicCouplesNow[i + 1], dynamicCouplesBoundaryNow[i + 1]);
      }
    }
    printf("\n");
  }

  int localDropletChangeCount = 0;

  int breakingCouplesCount = 0;
  int breakingCouplesBoundaryCount = 0;
  for(int i=0; i<dynamicCouplesNumberPast; i+=2){
    if(dynamicCouplesPast[i] > 0){
      if(h_isoCount[h_iso[dynamicCouplesPast[i]]] == 1){

	if(debug == DE_DEBUG){
	  printf("No Local bubbles # change: %d[%d[%d]]: %d\n",
		 i, dynamicCouplesPast[i],
		 h_iso[dynamicCouplesPast[i]],
		 h_isoCount[h_iso[dynamicCouplesPast[i]]]);
	}
	memset(localNumbChangesP0, 0, MAXNBUBBLE*sizeof(int));
	memset(localNumbChangesP1, 0, MAXNBUBBLE*sizeof(int));
	memset(localNumbChangesN0, 0, MAXNBUBBLE*sizeof(int));
	memset(localNumbChangesN1, 0, MAXNBUBBLE*sizeof(int));
	int localNumbChangesP0Count = 0;
	int localNumbChangesP1Count = 0;
	int localNumbChangesN0Count = 0;
	int localNumbChangesN1Count = 0;
   
	int test = TRUE;
	int testP0 = TRUE, testP1 = TRUE;
	int testN0 = TRUE, testN1 = TRUE;
	for(int j=0; j<linksCompressedSizeNow && test; j++){
	  int index = j + linksCompressedSizeNow*h_iso[dynamicCouplesPast[i]];
	  if(h_compressedLinksNow[index] == h_iso[dynamicCouplesPast[i + 1]])  
	    test = FALSE;
	  if(h_isoCount[h_iso[dynamicCouplesPast[i + 1]]] != 1){
	    test = FALSE;
	    localDropletChangeCount++;

	    if(debug == DE_DEBUG){
	      printf("Local bubbles # change [!!!]: %d[%d[%d]]: %d\n",
		     i + 1, dynamicCouplesPast[i + 1],
		     h_iso[dynamicCouplesPast[i + 1]], 
		     h_isoCount[h_iso[dynamicCouplesPast[i + 1]]]);    
	    }
	  }
	  
	  int indexPast0 = 
	    j + linksCompressedSizePast*dynamicCouplesPast[i];
	  int indexPast1 = 
	    j + linksCompressedSizePast*dynamicCouplesPast[i + 1];
	  int indexP2N0 = 
	    j + linksCompressedSizeNow*h_iso[dynamicCouplesPast[i]];
	  int indexP2N1 = 
	    j + linksCompressedSizeNow*h_iso[dynamicCouplesPast[i + 1]];

	  if(h_compressedLinksPast[indexPast0] < MAXLINK &&
	     h_isoCount[h_compressedLinksPast[indexPast0]] != 1){
	    testP0 = FALSE;
	    localNumbChangesP0[localNumbChangesP0Count] = 
	      h_compressedLinksPast[indexPast0];
	    localNumbChangesP0Count++;
	  }
	     
	  if(h_compressedLinksPast[indexPast1] < MAXLINK &&
	     h_isoCount[h_compressedLinksPast[indexPast1]] != 1){
	    testP1 = FALSE;
	    localNumbChangesP1[localNumbChangesP1Count] = 
	      h_compressedLinksPast[indexPast1];
	    localNumbChangesP1Count++;
	  }

	  if(h_compressedLinksNow[indexP2N0] < MAXLINK &&
	     h_isoCount[h_compressedLinksNow[indexP2N0]] != 1){
	    testN0 = FALSE;
	    localNumbChangesN0[localNumbChangesN0Count] = 
	      h_compressedLinksNow[indexP2N0];
	    localNumbChangesN0Count++;
	  }

	  if(h_compressedLinksNow[indexP2N1] < MAXLINK &&
	     h_isoCount[h_compressedLinksNow[indexP2N1]] != 1){
	    testN1 = FALSE;
	    localNumbChangesN1[localNumbChangesN1Count] = 
	      h_compressedLinksNow[indexP2N1];
	    localNumbChangesN1Count++;
	  }
	}

	localDropletChangeCount += (testP0 == FALSE || testP1 == FALSE ||
				    testN0 == FALSE || testN1 == FALSE);

	int testChangeCountP = TRUE;
	if(testP0 == FALSE || testP1 == FALSE){
	  for(int changeCount0 = 0; 
	      changeCount0 < localNumbChangesP0Count; changeCount0++){
	    for(int changeCount1 = 0; 
		changeCount1 < localNumbChangesP1Count; changeCount1++){
	      if(localNumbChangesP0[changeCount0] == 
		 localNumbChangesP1[changeCount1]){
		testChangeCountP = FALSE;
		
	      }
	    }
	  }
	}

	int testChangeCountN = TRUE;
	if(testN0 == FALSE || testN1 == FALSE){
	  for(int changeCount0 = 0; 
	      changeCount0 < localNumbChangesN0Count; changeCount0++){
	    for(int changeCount1 = 0; 
		changeCount1 < localNumbChangesN1Count; changeCount1++){
	      if(localNumbChangesN0[changeCount0] == 
		 localNumbChangesN1[changeCount1]){
		testChangeCountN = FALSE;
		
	      }
	    }
	  }
	}	  

	if(debug == DE_DEBUG){
	  printf("P {%d, %d}: %d %d %d %d %d %d %d\n", 
		 dynamicCouplesPast[i], dynamicCouplesPast[i + 1],
		 test, testChangeCountP, testChangeCountN,
		 testP0, testP1, testN0, testN1);
	}

	if(test == TRUE && 
	   (testChangeCountP == TRUE && testChangeCountN == TRUE)){

	  if(dynamicCouplesBoundaryPast[i] == 1 && 
	     dynamicCouplesBoundaryPast[i + 1] == 1 &&
	     isCoupleOnBoundary(dynamicCouplesPast[i], 
				dynamicCouplesPast[i + 1],
				h_compressedLinksPast, 
				linksCompressedSizePast)){
	    breakingCouplesBoundary[breakingCouplesBoundaryCount] = 
	      dynamicCouplesPast[i];
	    breakingCouplesBoundary[breakingCouplesBoundaryCount + 1] = 
	      dynamicCouplesPast[i + 1];
	    breakingCouplesBoundaryCount += 2;
	  }else{
	    breakingCouples[breakingCouplesCount] = dynamicCouplesPast[i];
	    breakingCouples[breakingCouplesCount + 1] = 
	      dynamicCouplesPast[i + 1];
	    breakingCouplesCount += 2;
	  }
	}      
      }else{
	localDropletChangeCount++;
	if(debug == DE_DEBUG){
	  printf("Local bubbles # change [!!!]: %d[%d[%d]]: %d\n",
		 i, dynamicCouplesPast[i],
		 h_iso[dynamicCouplesPast[i]], 
		 h_isoCount[h_iso[dynamicCouplesPast[i]]]);      	  
	}
      }
    }
  }

  int arisingCouplesCount = 0;
  int arisingCouplesBoundaryCount = 0;
  for(int i=0; i<dynamicCouplesNumberNow; i+=2){
    if(dynamicCouplesNow[i] > 0){
      if(h_isoCount[h_iso[dynamicCouplesNow[i]]] == 1){
	if(debug == DE_DEBUG){
	  printf("No Local bubbles # change: %d[%d]: %d\n",
		 i, dynamicCouplesNow[i],
		 h_isoCount[h_iso[dynamicCouplesNow[i]]]);
	}

	memset(localNumbChangesP0, 0, MAXNBUBBLE*sizeof(int));
	memset(localNumbChangesP1, 0, MAXNBUBBLE*sizeof(int));
	memset(localNumbChangesN0, 0, MAXNBUBBLE*sizeof(int));
	memset(localNumbChangesN1, 0, MAXNBUBBLE*sizeof(int));
	int localNumbChangesP0Count = 0;
	int localNumbChangesP1Count = 0;
	int localNumbChangesN0Count = 0;
	int localNumbChangesN1Count = 0;

	int test = TRUE;
	int testP0 = TRUE, testP1 = TRUE;
	int testN0 = TRUE, testN1 = TRUE;
	for(int j=0; j<linksCompressedSizePast && test; j++){
	  int index = j + linksCompressedSizePast*dynamicCouplesNow[i];
	  if(h_compressedLinksPast[index] == h_iso[dynamicCouplesNow[i + 1]])
	    test = FALSE;
	  if(h_isoCount[h_iso[dynamicCouplesNow[i + 1]]] != 1){
	    test = FALSE;
	    localDropletChangeCount++;
	    if(debug == DE_DEBUG){
	      printf("Local bubbles # change [!!!]: %d[%d]: %d\n",
		     i + 1, dynamicCouplesNow[i + 1],
		     h_isoCount[h_iso[dynamicCouplesNow[i + 1]]]);	    
	    }
	  }

	  int indexNow0 = 
	    j + linksCompressedSizeNow*h_iso[dynamicCouplesNow[i]];
	  int indexNow1 = 
	    j + linksCompressedSizeNow*h_iso[dynamicCouplesNow[i + 1]];
	  int indexN2P0 = 
	    j + linksCompressedSizePast*dynamicCouplesNow[i];
	  int indexN2P1 = 
	    j + linksCompressedSizePast*dynamicCouplesNow[i + 1];

	  if(h_compressedLinksPast[indexN2P0] < MAXLINK &&
	     h_isoCount[h_compressedLinksPast[indexN2P0]] != 1){
	    testP0 = FALSE;
	    localNumbChangesP0[localNumbChangesP0Count] = 
	      h_compressedLinksPast[indexN2P0];
	    localNumbChangesP0Count++;
	  }
	     
	  if(h_compressedLinksPast[indexN2P1] < MAXLINK &&
	     h_isoCount[h_compressedLinksPast[indexN2P1]] != 1){
	    testP1 = FALSE;
	    localNumbChangesP1[localNumbChangesP1Count] = 
	      h_compressedLinksPast[indexN2P1];
	    localNumbChangesP1Count++;
	  }
	  
	  if(h_compressedLinksNow[indexNow0] < MAXLINK &&
	     h_isoCount[h_compressedLinksNow[indexNow0]] != 1){
	    testN0 = FALSE;
	    localNumbChangesN0[localNumbChangesN0Count] = 
	      h_compressedLinksNow[indexNow0];
	    localNumbChangesN0Count++;
	  }

	  if(h_compressedLinksNow[indexNow1] < MAXLINK &&
	     h_isoCount[h_compressedLinksNow[indexNow1]] != 1){
	    testN1 = FALSE;
	    localNumbChangesN0[localNumbChangesN1Count] = 
	      h_compressedLinksNow[indexNow1];
	    localNumbChangesN1Count++;
	  }
	}

	localDropletChangeCount += (testP0 == FALSE || testP1 == FALSE ||
				    testN0 == FALSE || testN1 == FALSE);

	int testChangeCountP = TRUE;
	if(testP0 == FALSE || testP1 == FALSE){
	  for(int changeCount0 = 0; 
	      changeCount0 < localNumbChangesP0Count; changeCount0++){
	    for(int changeCount1 = 0; 
		changeCount1 < localNumbChangesP1Count; changeCount1++){
	      if(localNumbChangesP0[changeCount0] == 
		 localNumbChangesP1[changeCount1]){
		testChangeCountP = FALSE;				
	      }
	    }
	  }
	}

	int testChangeCountN = TRUE;
	if(testN0 == FALSE || testN1 == FALSE){
	  for(int changeCount0 = 0; 
	      changeCount0 < localNumbChangesN0Count; changeCount0++){
	    for(int changeCount1 = 0; 
		changeCount1 < localNumbChangesN1Count; changeCount1++){
	      if(localNumbChangesN0[changeCount0] == 
		 localNumbChangesN1[changeCount1]){
		testChangeCountN = FALSE;		
	      }
	    }
	  }
	}	  

	if(debug == DE_DEBUG){
	  printf("N {%d, %d}: %d %d %d %d %d %d %d\n", 
		 h_iso[dynamicCouplesNow[i]], h_iso[dynamicCouplesNow[i + 1]],
		 test, testChangeCountP, testChangeCountN,
		 testP0, testP1, testN0, testN1);
	}

	if(test == TRUE &&
	   testChangeCountP == TRUE && testChangeCountN == TRUE){

	  if(dynamicCouplesBoundaryNow[i] == 1 && 
	     dynamicCouplesBoundaryNow[i + 1] == 1 &&
	     isCoupleOnBoundary(h_iso[dynamicCouplesNow[i]], 
				h_iso[dynamicCouplesNow[i + 1]],
				h_compressedLinksNow, 
				linksCompressedSizeNow)){
	    arisingCouplesBoundary[arisingCouplesBoundaryCount] = 
	      dynamicCouplesNow[i];

	    arisingCouplesBoundary[arisingCouplesBoundaryCount + 1] = 
	      dynamicCouplesNow[i + 1];

	    arisingCouplesBoundaryCount += 2;	  
	  }else{
	    arisingCouples[arisingCouplesCount] = dynamicCouplesNow[i];
	    arisingCouples[arisingCouplesCount + 1] = dynamicCouplesNow[i + 1];
	    arisingCouplesCount += 2;
	  }
	}
      }else{
	localDropletChangeCount++;
	if(debug == DE_DEBUG){
	  printf("Local bubbles # change [!!!]: %d[%d]: %d\n",
		 i, dynamicCouplesNow[i],
		 h_isoCount[h_iso[dynamicCouplesNow[i]]]);      
	}	
      }
    }
  }
  
  if(debug == DE_DEBUG){
    printf("breaking couples: ");
    for(int i=0; i<breakingCouplesCount; i+=2){
      printf("{%04d, %04d} ", breakingCouples[i], breakingCouples[i + 1]);
    }
    printf("\n");
    printf("breaking couples boundary: ");
    for(int i=0; i<breakingCouplesBoundaryCount; i+=2){
      printf("{%04d, %04d} ", 
	     breakingCouplesBoundary[i], breakingCouplesBoundary[i + 1]);
    }
    printf("\n");
    printf("arising couples: ");
    for(int i=0; i<arisingCouplesCount; i+=2){
      printf("{%04d, %04d} ", 
	     h_iso[arisingCouples[i]], h_iso[arisingCouples[i + 1]]);
    }
    printf("\n");
    printf("arising couples boundary: ");
    for(int i=0; i<arisingCouplesBoundaryCount; i+=2){
      printf("{%04d, %04d} ", 
	     h_iso[arisingCouplesBoundary[i]], 
	     h_iso[arisingCouplesBoundary[i + 1]]);
    }
    printf("\n");
  }

  if(breakingCouplesCount == 0 && arisingCouplesCount == 0 &&
     breakingCouplesBoundaryCount == 0 && arisingCouplesBoundaryCount == 0 &&
     localDropletChangeCount == 0){
    fprintf(stderr, "Bad story...\n");
    return DE_EXIT_FAILURE;
  }

  if(breakingCouplesCount > 0){
    FILE *breakingLinks = fopen(breakingLinksName, "a");
    for(int i=0; i<breakingCouplesCount; i+=2){

      int indexLow = uniqueLabelsM1[h_iso[breakingCouples[i]]];
      int indexHigh = uniqueLabelsM1[h_iso[breakingCouples[i + 1]]];
      if(indexLow > indexHigh){
	int swap = indexLow;
	indexLow = indexHigh;
	indexHigh = swap;
      }
      int indexUniqueCouples = indexLow + indexHigh*(MAXNBUBBLE + 1);
      uniqueCouplesBreaking[indexUniqueCouples] += 1;

      int deltaTime = time - uniqueCouplesBreakingTime[indexUniqueCouples];
      uniqueCouplesBreakingTime[indexUniqueCouples] = time;

      fprintf(breakingLinks, 
	      "%010d %010d %010d %010d %14.7e %14.7e %010d %010d %010d %010d\n"
	      "%010d %010d %010d %010d %14.7e %14.7e %010d %010d %010d %010d\n\n", 
	      time, 
	      h_compressedHistogramPast[breakingCouples[i]],
	      h_linksCountNow[h_iso[breakingCouples[i]]],
	      h_linksCountPast[breakingCouples[i]], 
	      h_compressedXlabPast[breakingCouples[i]], 
	      h_compressedYlabPast[breakingCouples[i]],
	      uniqueCouplesBreaking[indexUniqueCouples],
	      indexUniqueCouples,
	      deltaTime,
	      uniqueLabelsM1[h_iso[breakingCouples[i]]],	      
	      time, 
	      h_compressedHistogramPast[breakingCouples[i + 1]],
	      h_linksCountNow[h_iso[breakingCouples[i + 1]]],
	      h_linksCountPast[breakingCouples[i + 1]], 
	      h_compressedXlabPast[breakingCouples[i + 1]], 
	      h_compressedYlabPast[breakingCouples[i + 1]],
	      uniqueCouplesBreaking[indexUniqueCouples],
	      indexUniqueCouples,
	      deltaTime,
	      uniqueLabelsM1[h_iso[breakingCouples[i + 1]]]);
      
    }
    fprintf(breakingLinks, "\n\n");
    fflush(breakingLinks);
    fclose(breakingLinks);
  }

  if(arisingCouplesCount > 0){
    FILE *arisingLinks = fopen(arisingLinksName, "a");
    for(int i=0; i<arisingCouplesCount; i+=2){

      int indexLow = uniqueLabelsM1[h_iso[arisingCouples[i]]];
      int indexHigh = uniqueLabelsM1[h_iso[arisingCouples[i + 1]]];
      if(indexLow > indexHigh){
	int swap = indexLow;
	indexLow = indexHigh;
	indexHigh = swap;
      }
      int indexUniqueCouples = indexLow + indexHigh*(MAXNBUBBLE + 1);
      uniqueCouplesArising[indexUniqueCouples] += 1;

      int deltaTime = time - uniqueCouplesArisingTime[indexUniqueCouples];
      uniqueCouplesArisingTime[indexUniqueCouples] = time;

      fprintf(arisingLinks, 
	      "%010d %010d %010d %010d %014.7e %014.7e %010d %010d %010d %010d\n"
	      "%010d %010d %010d %010d %014.7e %014.7e %010d %010d %010d %010d\n\n", 
	      time, 
	      h_compressedHistogramNow[h_iso[arisingCouples[i]]],
	      h_linksCountPast[arisingCouples[i]], 
	      h_linksCountNow[h_iso[arisingCouples[i]]], 
	      h_compressedXlabPast[arisingCouples[i]], 
	      h_compressedYlabNow[h_iso[arisingCouples[i]]],
	      uniqueCouplesArising[indexUniqueCouples],
	      indexUniqueCouples,
	      deltaTime,
	      uniqueLabelsM1[h_iso[arisingCouples[i]]],	      
	      time, 
	      h_compressedHistogramNow[h_iso[arisingCouples[i + 1]]],
	      h_linksCountPast[arisingCouples[i + 1]],
	      h_linksCountNow[h_iso[arisingCouples[i + 1]]], 
	      h_compressedXlabNow[h_iso[arisingCouples[i + 1]]], 
	      h_compressedYlabNow[h_iso[arisingCouples[i + 1]]],
	      uniqueCouplesArising[indexUniqueCouples],
	      indexUniqueCouples,
	      deltaTime,
	      uniqueLabelsM1[h_iso[arisingCouples[i + 1]]]);
    }
    fprintf(arisingLinks, "\n\n");
    fflush(arisingLinks);
    fclose(arisingLinks);
  }

  if(breakingCouplesBoundaryCount > 0){
    FILE *breakingLinksBoundary = fopen(breakingLinksBoundaryName, "a");
    for(int i=0; i<breakingCouplesBoundaryCount; i+=2){

      int indexLow = uniqueLabelsM1[h_iso[breakingCouplesBoundary[i]]];
      int indexHigh = uniqueLabelsM1[h_iso[breakingCouplesBoundary[i + 1]]];
      if(indexLow > indexHigh){
	int swap = indexLow;
	indexLow = indexHigh;
	indexHigh = swap;
      }
      int indexUniqueCouples = indexLow + indexHigh*(MAXNBUBBLE + 1);
      uniqueCouplesBreaking[indexUniqueCouples] += 1;

      int deltaTime = time - uniqueCouplesBreakingTime[indexUniqueCouples];
      uniqueCouplesBreakingTime[indexUniqueCouples] = time;

      fprintf(breakingLinksBoundary, 
	      "%010d %010d %010d %010d %14.7e %14.7e %010d %010d %010d %010d\n"
	      "%010d %010d %010d %010d %14.7e %14.7e %010d %010d %010d %010d\n\n", 
	      time, 
	      h_compressedHistogramPast[breakingCouplesBoundary[i]],
	      h_linksCountNow[h_iso[breakingCouplesBoundary[i]]],
	      h_linksCountPast[breakingCouplesBoundary[i]], 
	      h_compressedXlabPast[breakingCouplesBoundary[i]], 
	      h_compressedYlabPast[breakingCouplesBoundary[i]],
	      uniqueCouplesBreaking[indexUniqueCouples],
	      indexUniqueCouples,
	      deltaTime,
	      uniqueLabelsM1[h_iso[breakingCouplesBoundary[i]]],
	      time, 
	      h_compressedHistogramPast[breakingCouplesBoundary[i + 1]],
	      h_linksCountNow[h_iso[breakingCouplesBoundary[i + 1]]],
	      h_linksCountPast[breakingCouplesBoundary[i + 1]], 
	      h_compressedXlabPast[breakingCouplesBoundary[i + 1]], 
	      h_compressedYlabPast[breakingCouplesBoundary[i + 1]],
	      uniqueCouplesBreaking[indexUniqueCouples],
	      indexUniqueCouples,
	      deltaTime,
	      uniqueLabelsM1[h_iso[breakingCouplesBoundary[i + 1]]]);
    }
    fprintf(breakingLinksBoundary, "\n\n");
    fflush(breakingLinksBoundary);
    fclose(breakingLinksBoundary);
  }

  if(arisingCouplesBoundaryCount > 0){
    FILE *arisingLinksBoundary = fopen(arisingLinksBoundaryName, "a");
    for(int i=0; i<arisingCouplesBoundaryCount; i+=2){

      int indexLow = uniqueLabelsM1[h_iso[arisingCouplesBoundary[i]]];
      int indexHigh = uniqueLabelsM1[h_iso[arisingCouplesBoundary[i + 1]]];
      if(indexLow > indexHigh){
	int swap = indexLow;
	indexLow = indexHigh;
	indexHigh = swap;
      }
      int indexUniqueCouples = indexLow + indexHigh*(MAXNBUBBLE + 1);
      uniqueCouplesArising[indexUniqueCouples] += 1;

      int deltaTime = time - uniqueCouplesArisingTime[indexUniqueCouples];
      uniqueCouplesArisingTime[indexUniqueCouples] = time;

      fprintf(arisingLinksBoundary, 
	      "%010d %010d %010d %010d %014.7e %014.7e\n"
	      "%010d %010d %010d %010d %014.7e %014.7e\n\n", 
	      time, 
	      h_compressedHistogramNow[h_iso[arisingCouplesBoundary[i]]],
	      h_linksCountPast[arisingCouplesBoundary[i]], 
	      h_linksCountNow[h_iso[arisingCouplesBoundary[i]]], 
	      h_compressedXlabPast[arisingCouplesBoundary[i]], 
	      h_compressedYlabNow[h_iso[arisingCouplesBoundary[i]]],
	      uniqueCouplesArising[indexUniqueCouples],
	      indexUniqueCouples,
	      deltaTime,
	      uniqueLabelsM1[h_iso[arisingCouplesBoundary[i]]],
	      time, 
	      h_compressedHistogramNow[h_iso[arisingCouplesBoundary[i + 1]]],
	      h_linksCountPast[arisingCouplesBoundary[i + 1]],
	      h_linksCountNow[h_iso[arisingCouplesBoundary[i + 1]]], 
	      h_compressedXlabNow[h_iso[arisingCouplesBoundary[i + 1]]], 
	      h_compressedYlabNow[h_iso[arisingCouplesBoundary[i + 1]]],
	      uniqueCouplesArising[indexUniqueCouples],
	      indexUniqueCouples,
	      deltaTime,
	      uniqueLabelsM1[h_iso[arisingCouplesBoundary[i + 1]]]);
    }
    fprintf(arisingLinksBoundary, "\n\n");
    fflush(arisingLinksBoundary);
    fclose(arisingLinksBoundary);
  }

  return 0;
}

extern REAL *readBinaryRho(char *fileName, int h_nx, int h_ny){
  int rhoSize = (h_nx + 2)*(h_ny + 2);
  REAL *rho = (REAL *)malloc(rhoSize*sizeof(REAL));
  if(rho == NULL){
    fprintf(stderr, "Horreo readBinaryRho\n");
    return NULL;
  }
  FILE *rhoInput = fopen(fileName, "rb");
  if(rhoInput == NULL){
    fprintf(stderr, "Horreo readBinaryRho\n");
    return NULL;
  }
  fread(rho, sizeof(REAL), rhoSize, rhoInput);
  fflush(rhoInput);
  fclose(rhoInput);

  return rho;
}

extern void makeDelaunayUnsortedV4(int grid, int block, int h_nla,
				   unsigned int *d_label, 
				   int *d_bubble,
				   int *d_flagObstacle,
				   int *d_links, 
				   int *d_vertexCount,
				   int *d_vertexCountSymm,
				   int *d_vertexMult,
				   float *d_xlab, float *d_ylab){

  device_function_computeDistancesV4<<<grid,block>>>
    (d_label, d_bubble, d_flagObstacle,
     d_xlab, d_ylab);
    
  MY_CUDA_CHECK( cudaMemset(d_links, 0x77, sizeof(int)*
			    MAXNBUBBLE*MAXNBUBBLE ) );
  
  MY_CUDA_CHECK( cudaMemset(d_vertexCount, 0, 
			    sizeof(int)*(MAXNBUBBLE + 1)) );

  MY_CUDA_CHECK( cudaMemset(d_vertexCountSymm, 0, 
			    sizeof(int)*(MAXNBUBBLE + 1)) );

  MY_CUDA_CHECK( cudaMemset(d_vertexMult, 0, 
			    sizeof(int)*h_nla) );

  device_function_linksWriterNew5<<<grid,block>>> 
    (d_label, d_links,
     d_vertexCount, d_vertexCountSymm, d_vertexMult,
     d_xlab, d_ylab, d_bubble);

  return;
}

extern void printIsoCMToFile(float *d_xlab1, float *d_ylab1,
			     float *d_xlab2, float *d_ylab2,
			     int *d_bubble1, int *d_bubble2,
			     int *d_iso, int h_nla, int time){

  float *h_xlab1 = (float *)malloc(h_nla*sizeof(float));
  float *h_ylab1 = (float *)malloc(h_nla*sizeof(float));
  float *h_xlab2 = (float *)malloc(h_nla*sizeof(float));
  float *h_ylab2 = (float *)malloc(h_nla*sizeof(float));

  MY_CUDA_CHECK( cudaMemcpy(h_xlab1, d_xlab1,
			    h_nla*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  MY_CUDA_CHECK( cudaMemcpy(h_ylab1, d_ylab1,
			    h_nla*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  MY_CUDA_CHECK( cudaMemcpy(h_xlab2, d_xlab2,
			    h_nla*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  MY_CUDA_CHECK( cudaMemcpy(h_ylab2, d_ylab2,
			    h_nla*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  int *h_bubble1 = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
  int *h_bubble2 = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));

  MY_CUDA_CHECK( cudaMemcpy(h_bubble1, d_bubble1,
			    (MAXNBUBBLE + 1)*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  MY_CUDA_CHECK( cudaMemcpy(h_bubble2, d_bubble2,
			    (MAXNBUBBLE + 1)*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  int *h_iso = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));

  MY_CUDA_CHECK( cudaMemcpy(h_iso, d_iso,
			    (MAXNBUBBLE + 1)*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  char outIsoCMName[1024];
  
  snprintf(outIsoCMName, sizeof(outIsoCMName), 
	   "newOutIsoCMTime%09d", time);

  FILE *outIsoCM = fopen(outIsoCMName, "w");

  for(int i=1; i<=h_bubble1[0]; i++){
    fprintf(outIsoCM, "%d %f %f\n%d %f %f\n\n\n",
	    i, h_xlab1[h_bubble1[i]], h_ylab1[h_bubble1[i]],
	    h_iso[i], 
	    h_xlab2[h_bubble2[h_iso[i]]], 
	    h_ylab2[h_bubble2[h_iso[i]]]);
  }

  fflush(outIsoCM);
  fclose(outIsoCM);

  free(h_iso);
  free(h_bubble1);
  free(h_bubble2);
  free(h_xlab2);
  free(h_ylab2);
  free(h_xlab1);
  free(h_ylab1);

  return;
}

extern void printAdjacencyToFile(float *d_xlab, float *d_ylab, 
				 int *d_bubble, int *d_links,
				 int h_nla, int time){

  float *h_xlab = (float *)malloc(h_nla*sizeof(float));
  float *h_ylab = (float *)malloc(h_nla*sizeof(float));
  int *h_bubble = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
  int *h_links = (int *)malloc(MAXNBUBBLE*MAXNBUBBLE*sizeof(int));

  MY_CUDA_CHECK( cudaMemcpy(h_xlab, d_xlab,
			    h_nla*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  MY_CUDA_CHECK( cudaMemcpy(h_ylab, d_ylab,
			    h_nla*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  MY_CUDA_CHECK( cudaMemcpy(h_bubble, d_bubble,
			    (MAXNBUBBLE + 1)*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  MY_CUDA_CHECK( cudaMemcpy(h_links, d_links,
			    MAXNBUBBLE*MAXNBUBBLE*sizeof(int),
			    cudaMemcpyDeviceToHost) );


  char outDelaunayName[1024];
  
  snprintf(outDelaunayName, sizeof(outDelaunayName), 
	   "newOutDelaunayTime%09d", time);

  FILE *outDelaunay = fopen(outDelaunayName, "w");
  for(int i=1; i<=h_bubble[0]; i++){
    int test = 1;
    for(int j=0; j<MAXNBUBBLE && test; j++){
      int index = i*MAXNBUBBLE + j;
      if(h_links[index] < h_bubble[0]){
	fprintf(outDelaunay, "%d %f %f\n", 
		i, h_xlab[h_bubble[i]], h_ylab[h_bubble[i]]);	
	fprintf(outDelaunay, "%d %f %f\n",
		h_links[index], h_xlab[h_bubble[h_links[index]]],
		h_ylab[h_bubble[h_links[index]]]);
	fprintf(outDelaunay, "\n");
      }else test = 0;
    }
    fprintf(outDelaunay, "\n\n");
  }
  fflush(outDelaunay);
  fclose(outDelaunay);

  free(h_xlab);
  free(h_ylab);
  free(h_links);
  free(h_bubble);

  return;
}

extern void findDropletsSimV4(int grid, int block, int h_nla, 
			      REAL *d_rho, int *d_flag, 
			      int *d_flagObstacle,
			      unsigned int *d_label,
			      int **d_histogram, int *d_bubble,
			      float *d_xlab, float *d_ylab, 
			      int k){
  static int first = TRUE, second = TRUE;

  device_function_set_flagSimObstacle<<<grid,block>>>
    (d_rho, d_flag, d_flagObstacle);
    
  labelling(h_nla, d_flag, d_label, block, grid);

  if(!second && (k & 1U)){
    MY_CUDA_CHECK( cudaMemset(*d_histogram, 0, sizeof(int)*h_nla) );
  }  
  if(!first && (k & 1U) == 0){
    MY_CUDA_CHECK( cudaMemset(*d_histogram, 0, sizeof(int)*h_nla) );
    second = FALSE;
  }
  
  *d_histogram = cntlabelbythrust(h_nla, d_label, k);
  
  device_function_findPeriodic<<<grid,block>>>(d_label, *d_histogram);
  
  MY_CUDA_CHECK( cudaMemset(d_bubble, 0, sizeof(int)*(MAXNBUBBLE + 1)
			    ) );
  MY_CUDA_CHECK( cudaMemset(d_xlab, 0, h_nla*sizeof(float) ) );
  MY_CUDA_CHECK( cudaMemset(d_ylab, 0, h_nla*sizeof(float) ) );
  
  device_function_compute_mc1New<<<grid,block>>>(d_label, 
						 *d_histogram,
						 d_xlab, d_ylab);
  
  device_function_compute_mc2New<<<grid,block>>>(*d_histogram, 
						 d_flag, 
						 d_xlab, d_ylab, 
						 d_bubble);
  first = FALSE;
  return;
}

extern void printDeviceIntArrayGridToFile(int *d_array, int size, 
					  int h_nx, int h_ny,
					  const char *fileName){

  int *h_array = (int *)malloc(size*sizeof(int));
  MY_CUDA_CHECK( cudaMemcpy(h_array, d_array,
			    size*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  FILE *out = fopen(fileName, "w");

  for(int y=0; y<h_ny; y++){
    for(int x=0; x<h_nx; x++){
      int index = x + y*h_nx;
      fprintf(out, "%d %d %d\n", x, y, h_array[index]);
    }
  }

  fflush(out);
  fclose(out);
  free(h_array);
  return;
}

extern void printDeviceIntArrayToFile(int *d_array, int size, 
				      const char *fileName){
  int *h_array = (int *)malloc(size*sizeof(int));
  MY_CUDA_CHECK( cudaMemcpy(h_array, d_array,
			    size*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  FILE *out = fopen(fileName, "w");

  for(int i=0; i<size; i++){
    fprintf(out, "%d %d\n", i, h_array[i]);
  }

  fflush(out);
  fclose(out);
  free(h_array);

  return;
}

extern void printDeviceInt2ArrayToFile(int *d_array1, int *d_array2,
				       int size, 
				       const char *fileName){
  int *h_array1 = (int *)malloc(size*sizeof(int));
  MY_CUDA_CHECK( cudaMemcpy(h_array1, d_array1,
			    size*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  int *h_array2 = (int *)malloc(size*sizeof(int));
  MY_CUDA_CHECK( cudaMemcpy(h_array1, d_array1,
			    size*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  FILE *out = fopen(fileName, "w");

  for(int i=0; i<size; i++){
    fprintf(out, "%d %d\n", i, h_array2[h_array1[i]]);
  }

  fflush(out);
  fclose(out);
  free(h_array1);
  free(h_array2);

  return;
}

extern void printDeviceCMToFile(int *d_array1, 
				float *d_array2, float *d_array3,
				int size1, int size2, int size3,
				const char *fileName){

  int *h_array1 = (int *)malloc(size1*sizeof(int));
  MY_CUDA_CHECK( cudaMemcpy(h_array1, d_array1,
			    size1*sizeof(int),
			    cudaMemcpyDeviceToHost) );

  float *h_array2 = (float *)malloc(size2*sizeof(float));
  MY_CUDA_CHECK( cudaMemcpy(h_array2, d_array2,
			    size2*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  float *h_array3 = (float *)malloc(size3*sizeof(float));
  MY_CUDA_CHECK( cudaMemcpy(h_array3, d_array3,
			    size3*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  FILE *out = fopen(fileName, "w");

  printf("h_array1[0]: %d\n", h_array1[0]);

  for(int i=1; i<=h_array1[0]; i++){
    fprintf(out, "%d %f %f\n", i, 
	    h_array2[h_array1[i]],
	    h_array3[h_array1[i]]);
  }

  fflush(out);
  fclose(out);
  free(h_array1);
  free(h_array2);
  free(h_array3);

  return;
}

extern void printDeviceUnsignedArrayToFile(unsigned int *d_array, 
					   int size, 
					   const char *fileName){
  unsigned int *h_array = 
    (unsigned int *)malloc(size*sizeof(unsigned int));

  MY_CUDA_CHECK( cudaMemcpy(h_array, d_array,
			    size*sizeof(unsigned int),
			    cudaMemcpyDeviceToHost) );

  FILE *out = fopen(fileName, "w");

  for(int i=0; i<size; i++){
    fprintf(out, "%d %d\n", i, h_array[i]);
  }

  fflush(out);
  fclose(out);
  free(h_array);

  return;
}


extern void printDeviceUnsignedArrayGridToFile(unsigned int *d_array, 
					       int size, 
					       int h_nx, int h_ny,
					       const char *fileName){

  unsigned int *h_array = 
    (unsigned int *)malloc(size*sizeof(unsigned int));
  MY_CUDA_CHECK( cudaMemcpy(h_array, d_array,
			    size*sizeof(unsigned int),
			    cudaMemcpyDeviceToHost) );

  FILE *out = fopen(fileName, "w");

  for(int y=0; y<h_ny; y++){
    for(int x=0; x<h_nx; x++){
      int index = x + y*h_nx;
      fprintf(out, "%d %d %d\n", x, y, h_array[index]);
    }
  }

  fflush(out);
  fclose(out);
  free(h_array);
  return;
}

extern void printDeviceREALArrayGridToFile(REAL *d_array, int size, 
					   int h_nx, int h_ny,
					   const char *fileName){

  REAL *h_array = (REAL *)malloc(size*sizeof(REAL));
  MY_CUDA_CHECK( cudaMemcpy(h_array, d_array,
			    size*sizeof(REAL),
			    cudaMemcpyDeviceToHost) );

  FILE *out = fopen(fileName, "w");

  for(int y=0; y<h_ny; y++){
    for(int x=0; x<h_nx; x++){
      int index = x + y*h_nx;
      fprintf(out, "%d %d %lf\n", x, y, h_array[index]);
    }
  }

  fflush(out);
  fclose(out);
  free(h_array);
  return;
}

extern void printDeviceFloatArrayGridToFile(float *d_array, int size, 
					   int h_nx, int h_ny,
					   const char *fileName){

  float *h_array = (float *)malloc(size*sizeof(float));
  MY_CUDA_CHECK( cudaMemcpy(h_array, d_array,
			    size*sizeof(float),
			    cudaMemcpyDeviceToHost) );

  FILE *out = fopen(fileName, "w");

  for(int y=0; y<h_ny; y++){
    for(int x=0; x<h_nx; x++){
      int index = x + y*h_nx;
      fprintf(out, "%d %d %f\n", x, y, h_array[index]);
    }
  }

  fflush(out);
  fclose(out);
  free(h_array);
  return;
}

extern void printHostREALArrayGridToFile(REAL *h_array, int size, 
					 int h_nx, int h_ny,
					 const char *fileName){

  FILE *out = fopen(fileName, "w");

  for(int x=0; x<h_nx; x++){
    for(int y=0; y<h_ny; y++){
      int index = x + y*h_nx;
      fprintf(out, "%d %d %lf\n", x, y, h_array[index]);
    }
  }

  fflush(out);
  fclose(out);
  return;
}

extern int delaunayBinaryDump(int *d_linksNow, int *d_linksPast, 
			      int *d_iso, 
			      int *d_bubbleNow, int *d_bubblePast,
			      int *d_histogramNow, int *d_histogramPast,
			      float *d_xlabNow, float *d_ylabNow,
			      float *d_xlabPast, float *d_ylabPast,
			      int h_nx, int h_ny, int h_nla, 
			      int time, cudaStream_t *stream){

  static int *h_linksNow = NULL, *h_linksPast = NULL, *h_iso = NULL;
  static float *h_xlabNow = NULL, *h_ylabNow = NULL;
  static float *h_xlabPast = NULL, *h_ylabPast = NULL;
  static int *h_bubbleNow = NULL, *h_bubblePast = NULL;
  static int *h_histogramNow = NULL, *h_histogramPast = NULL;
  static int maxNBubble = MAXNBUBBLE;

  static char delaunayNowName[MAXFILENAME];
  snprintf(delaunayNowName, sizeof(delaunayNowName), 
	   "delaunayNowTime%09d", time);

  static char delaunayPastName[MAXFILENAME];
  snprintf(delaunayPastName, sizeof(delaunayPastName), 
	   "delaunayPastTime%09d", time);

  static char isoName[MAXFILENAME];
  snprintf(isoName, sizeof(isoName), 
	   "isoTime%09d", time);

  if(h_linksNow == NULL){
    h_linksNow = (int *)malloc(MAXNBUBBLE*MAXNBUBBLE*sizeof(int));
    h_linksPast = (int *)malloc(MAXNBUBBLE*MAXNBUBBLE*sizeof(int));
    h_iso = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
    h_bubbleNow = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
    h_bubblePast = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
    h_histogramNow = (int *)malloc(h_nla*sizeof(int));
    h_histogramPast = (int *)malloc(h_nla*sizeof(int));
    h_xlabNow = (float *)malloc(h_nla*sizeof(float));
    h_ylabNow = (float *)malloc(h_nla*sizeof(float));
    h_xlabPast = (float *)malloc(h_nla*sizeof(float));
    h_ylabPast = (float *)malloc(h_nla*sizeof(float));

    if(h_linksNow == NULL || h_linksPast == NULL || h_iso == NULL ||
       h_bubbleNow == NULL || h_bubblePast == NULL || 
       h_histogramNow == NULL || h_histogramPast == NULL ||
       h_xlabNow == NULL || h_ylabNow == NULL ||
       h_xlabPast == NULL || h_ylabPast == NULL){

      fprintf(stderr, "Horreo!\n");
      return 1;
    }
  }

  MY_CUDA_CHECK( cudaMemcpyAsync( h_linksNow, d_linksNow, 
				  MAXNBUBBLE*MAXNBUBBLE*sizeof(int),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_linksPast, d_linksPast, 
				  MAXNBUBBLE*MAXNBUBBLE*sizeof(int),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_iso, d_iso, 
				  (MAXNBUBBLE + 1)*sizeof(int),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_bubbleNow, d_bubbleNow, 
				  (MAXNBUBBLE + 1)*sizeof(int),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_bubblePast, d_bubblePast, 
				  (MAXNBUBBLE + 1)*sizeof(int),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_histogramNow, d_histogramNow, 
				  h_nla*sizeof(int),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_histogramPast, d_histogramPast, 
				  h_nla*sizeof(int),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_xlabNow, d_xlabNow, 
				  h_nla*sizeof(float),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_ylabNow, d_ylabNow, 
				  h_nla*sizeof(float),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_xlabPast, d_xlabPast, 
				  h_nla*sizeof(float),
				  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK( cudaMemcpyAsync( h_ylabPast, d_ylabPast, 
				  h_nla*sizeof(float),
				  cudaMemcpyDeviceToHost, stream[0] ) );


  MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));

  FILE *outDelaunayNow = fopen(delaunayNowName, "wb");

  fwrite(&h_nx, sizeof(int), 1, outDelaunayNow);
  fwrite(&h_ny, sizeof(int), 1, outDelaunayNow);
  fwrite(&maxNBubble, sizeof(int), 1, outDelaunayNow);
  fwrite(h_bubbleNow, sizeof(int), (MAXNBUBBLE + 1), outDelaunayNow);
  fwrite(h_histogramNow, sizeof(int), h_nla, outDelaunayNow);
  fwrite(h_linksNow, sizeof(int), MAXNBUBBLE*MAXNBUBBLE, 
	 outDelaunayNow);
  fwrite(h_xlabNow, sizeof(float), h_nla, outDelaunayNow);
  fwrite(h_ylabNow, sizeof(float), h_nla, outDelaunayNow);

  fflush(outDelaunayNow);
  fclose(outDelaunayNow);

  FILE *outDelaunayPast = fopen(delaunayPastName, "wb");

  fwrite(&h_nx, sizeof(int), 1, outDelaunayPast);
  fwrite(&h_ny, sizeof(int), 1, outDelaunayPast);
  fwrite(&maxNBubble, sizeof(int), 1, outDelaunayPast);
  fwrite(h_bubblePast, sizeof(int), (MAXNBUBBLE + 1), outDelaunayPast);
  fwrite(h_histogramPast, sizeof(int), h_nla, outDelaunayPast);
  fwrite(h_linksPast, sizeof(int), MAXNBUBBLE*MAXNBUBBLE, 
	 outDelaunayPast);
  fwrite(h_xlabPast, sizeof(float), h_nla, outDelaunayPast);
  fwrite(h_ylabPast, sizeof(float), h_nla, outDelaunayPast);

  fflush(outDelaunayPast);
  fclose(outDelaunayPast);

  FILE *outIso = fopen(isoName, "wb");
  fwrite(h_iso, sizeof(int), (MAXNBUBBLE + 1), outIso);
  fflush(outIso);
  fclose(outIso);

  return 0;
}

extern int delaunayTriggerV4(REAL *d_rho, int *d_flagObstacle, 
			     int xSize, int ySize, 
			     REAL h_threshold, int time, 
			     int whichGPU,
			     cudaStream_t *stream, 
			     int pbcx, int pbcy, 
			     int debug){
  static ContextPtr context;
  static bool first = TRUE;
  static int counter = 0;
  static int h_nx = 0, h_ny = 0, h_nla = 0;
  static int h_hnx = 0, h_hny = 0;
  
  static int *d_flag = NULL, *d_bubble1 = NULL, *d_bubble2 = NULL;
  static unsigned int* d_label = NULL;
  static int *d_histogram1 = NULL;
  static int *d_histogram2 = NULL;
  
  static float *d_xlab1 = NULL, *d_ylab1 = NULL;
  static float *d_xlab2 = NULL, *d_ylab2 = NULL;
  
  static int *d_vertexCount = NULL, *d_vertexCountSymm = NULL;
  static int *d_vertexMult = NULL;
  
  static int block = THREADS, grid = 0;
  static int blockUnique = THREADS;
  static int gridUnique = (MAXNBUBBLE + blockUnique - 1)/blockUnique;
  
  static int blockTr = 2*THREADS;
  static int gridTr = MAXNBUBBLE;
  
  static int *d_links1 = NULL, *d_links2 = NULL, *d_linksSwap = NULL;
  static int *d_sortKeys = NULL;
  
  static int *d_iso = NULL, *d_isoCount = NULL;
  static int *d_trigger = NULL, *d_chkInvert = NULL;
  
  static int nBubbles1 = 0, nBubbles2 = 0;
  static FILE *nBubblesOut = NULL, *nLinksOut = NULL;
  
  static int *d_linksCountNow = NULL, *d_linksCountPast = NULL;
  static int *h_linksCountNow = NULL, *h_linksCountPast = NULL;
  static int *d_localTrigger = NULL;
  static int *h_localTrigger = NULL;
  
  static int maxLinksCountNow = 0, maxLinksCountPast = 0;
  static int maxLinksCountAbsNow = 0;
  static int sumLinksNow = 0, sumLinksPast = 0;
  static int linksCompressedSize = 0;

  static float *d_compressedXlabNow = NULL, *d_compressedYlabNow = NULL;
  static float *d_compressedXlabPast = NULL, *d_compressedYlabPast = NULL;
  
  static float *h_compressedXlabNow = NULL, *h_compressedYlabNow = NULL;
  static float *h_compressedXlabPast = NULL, *h_compressedYlabPast = NULL;
  
  static int *d_compressedHistogramNow = NULL;
  static int *d_compressedHistogramPast = NULL;
  static int *h_compressedHistogramNow = NULL;
  static int *h_compressedHistogramPast = NULL;
  
  static int *h_iso = NULL, *h_isoCount = NULL, *h_isoM1 = NULL;
  static int *uniqueLabels = NULL, *uniqueLabelsM1 = NULL;
  static int *uniqueCouplesBreaking = NULL;
  static int *uniqueCouplesArising = NULL;
  static int *uniqueCouplesBreakingTime = NULL;
  static int *uniqueCouplesArisingTime = NULL;
  static char outDirTrigger[MAXDIRNAME];
  static char outDirNoTrigger[MAXDIRNAME];
  static char nBubblesName[MAXFILENAME];
  static char nLinksName[MAXFILENAME];
  
  int h_trigger = 0, h_chkInvert = 0;
  
  if(first){
    snprintf(outDirTrigger, sizeof(outDirTrigger), "delaunayTriggerDir/");
    snprintf(outDirNoTrigger, sizeof(outDirNoTrigger), 
	     "delaunayNoTriggerDir/");
    char cmd[MAXCMD];
    snprintf(cmd, sizeof(cmd), 
	     "mkdir %s; mkdir %s\n", outDirTrigger, outDirNoTrigger);
    system(cmd);
    
    context = CreateCudaDevice(whichGPU);
    MY_CUDA_CHECK( cudaGetSymbolAddress((void **)&d_trigger, trigger) );
    MY_CUDA_CHECK( cudaGetSymbolAddress((void **)&d_chkInvert, chkInvert) );
    
    snprintf(nBubblesName, sizeof(nBubblesName), "%s/nBubblesOut",
	     outDirTrigger);
    nBubblesOut = fopen(nBubblesName, "a");
    snprintf(nLinksName, sizeof(nLinksName), "%s/nLinksOut",
	     outDirTrigger);
    nLinksOut = fopen(nLinksName, "a");
    
    int *h_sortKeys = (int *)malloc(MAXNBUBBLE*sizeof(int));
    MY_CUDA_CHECK( MallocCuda((void**) &d_sortKeys, MAXNBUBBLE*sizeof(int) ) );
    for(int i=0; i<MAXNBUBBLE; i++) h_sortKeys[i] = i*MAXNBUBBLE;
    MY_CUDA_CHECK( cudaMemcpy(d_sortKeys, h_sortKeys,
			      MAXNBUBBLE*sizeof(int),
			      cudaMemcpyHostToDevice) );
    free(h_sortKeys);
    
    h_nx = xSize; h_ny = ySize;
    h_hnx = h_nx/2; h_hny = h_ny/2;
    h_nla = xSize*ySize;
    
    grid = (h_nla + block - 1)/block;
    
    MY_CUDA_CHECK( cudaMemcpyToSymbol(nx, &h_nx, sizeof(int), 0,
				      cudaMemcpyHostToDevice) );
    
    MY_CUDA_CHECK( cudaMemcpyToSymbol(ny, &h_ny, sizeof(int), 0,
				      cudaMemcpyHostToDevice) );
    
    MY_CUDA_CHECK( cudaMemcpyToSymbol(hnx, &h_hnx, sizeof(int), 0,
				      cudaMemcpyHostToDevice) );
    
    MY_CUDA_CHECK( cudaMemcpyToSymbol(hny, &h_hny, sizeof(int), 0,
				      cudaMemcpyHostToDevice) );
    
    MY_CUDA_CHECK( cudaMemcpyToSymbol(nla, &h_nla, sizeof(int), 0,
				      cudaMemcpyHostToDevice) );
    
    MY_CUDA_CHECK( cudaMemcpyToSymbol(bst, &h_bst, sizeof(int), 0,
				      cudaMemcpyHostToDevice) );
    
    MY_CUDA_CHECK( cudaMemcpyToSymbol(threshold, &h_threshold, 
				      sizeof(REAL), 0,
				      cudaMemcpyHostToDevice) );

    MY_CUDA_CHECK( cudaMemcpyToSymbol(d_pbcx, &pbcx, 
				      sizeof(int), 0,
				      cudaMemcpyHostToDevice) );

    MY_CUDA_CHECK( cudaMemcpyToSymbol(d_pbcy, &pbcy, 
				      sizeof(int), 0,
				      cudaMemcpyHostToDevice) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_flag, h_nla*sizeof(int) ) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_label, 
			      h_nla*sizeof(unsigned int) ) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_xlab1, h_nla*sizeof(float) ) );
    MY_CUDA_CHECK( MallocCuda((void**) &d_ylab1, h_nla*sizeof(float) ) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_xlab2, h_nla*sizeof(float) ) );
    MY_CUDA_CHECK( MallocCuda((void**) &d_ylab2, h_nla*sizeof(float) ) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_bubble1, 
			      sizeof(int)*(MAXNBUBBLE + 1) ) );
    MY_CUDA_CHECK( MallocCuda((void**) &d_bubble2, 
			      sizeof(int)*(MAXNBUBBLE + 1) ) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_linksCountNow, 
			      sizeof(int)*(MAXNBUBBLE + 1) ) );
    MY_CUDA_CHECK( MallocCuda((void**) &d_linksCountPast, 
			      sizeof(int)*(MAXNBUBBLE + 1) ) );
    MY_CUDA_CHECK( MallocCuda((void**) &d_localTrigger, 
			      sizeof(int)*(MAXNBUBBLE + 1) ) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_iso, 
			      sizeof(int)*(MAXNBUBBLE + 1) ) );
    MY_CUDA_CHECK( MallocCuda((void**) &d_isoCount, 
			      sizeof(int)*(MAXNBUBBLE + 1) ) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_vertexCount, 
			      sizeof(int)*(MAXNBUBBLE + 1)) );

    MY_CUDA_CHECK( MallocCuda((void**) &d_vertexCountSymm, 
			      sizeof(int)*(MAXNBUBBLE + 1)) );

    MY_CUDA_CHECK( MallocCuda((void**) &d_vertexMult, 
			      sizeof(int)*h_nla) );
    
    MY_CUDA_CHECK( MallocCuda((void**) &d_links1, 
			      sizeof(int)*MAXNBUBBLE*MAXNBUBBLE ) );
    MY_CUDA_CHECK( MallocCuda((void**) &d_links2, 
			      sizeof(int)*MAXNBUBBLE*MAXNBUBBLE ) );
    MY_CUDA_CHECK( MallocCuda((void**) &d_linksSwap, 
			      sizeof(int)*MAXNBUBBLE*MAXNBUBBLE ) );
    
    h_linksCountNow = (int *)malloc(MAXNBUBBLE*sizeof(int));
    h_linksCountPast = (int *)malloc(MAXNBUBBLE*sizeof(int));

    if(h_linksCountNow == NULL || h_linksCountPast == NULL){
      fprintf(stderr, "No Alloc h_linksCountNow/Past\n");
      return DE_EXIT_FAILURE;
    }
    
    h_compressedXlabNow = (float *)malloc(MAXNBUBBLE*sizeof(float));
    h_compressedYlabNow = (float *)malloc(MAXNBUBBLE*sizeof(float));
    
    h_compressedXlabPast = (float *)malloc(MAXNBUBBLE*sizeof(float));
    h_compressedYlabPast = (float *)malloc(MAXNBUBBLE*sizeof(float));

    if(h_compressedXlabNow == NULL || h_compressedYlabNow == NULL ||
       h_compressedXlabPast == NULL || h_compressedYlabPast == NULL){
      fprintf(stderr, "No Alloc h_compressedX/YlabNow/Past\n");
      return DE_EXIT_FAILURE;
    }
    
    MY_CUDA_CHECK( MallocCuda((void **) &d_compressedXlabNow, 
			      MAXNBUBBLE*sizeof(float) ) );
    MY_CUDA_CHECK( MallocCuda((void **) &d_compressedYlabNow, 
			      MAXNBUBBLE*sizeof(float) ) );
    MY_CUDA_CHECK( MallocCuda((void **) &d_compressedXlabPast, 
			      MAXNBUBBLE*sizeof(float) ) );
    MY_CUDA_CHECK( MallocCuda((void **) &d_compressedYlabPast, 
			      MAXNBUBBLE*sizeof(float) ) );
    
    h_compressedHistogramNow = (int *)malloc(MAXNBUBBLE*sizeof(int));
    h_compressedHistogramPast = (int *)malloc(MAXNBUBBLE*sizeof(int));
    h_iso = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
    h_isoCount = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
    h_isoM1 = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));

    uniqueLabels = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
    uniqueLabelsM1 = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
    uniqueCouplesBreaking = (int *)malloc((MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*
					  sizeof(int));
    uniqueCouplesArising = (int *)malloc((MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*
					 sizeof(int));
    uniqueCouplesBreakingTime = (int *)malloc((MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*
					      sizeof(int));
    uniqueCouplesArisingTime = (int *)malloc((MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*
					     sizeof(int));

    memset(uniqueCouplesBreaking, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
    memset(uniqueCouplesArising, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
    memset(uniqueCouplesBreakingTime, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
    memset(uniqueCouplesArisingTime, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));

    for(int i=0; i<MAXNBUBBLE + 1; i++) uniqueLabels[i] = i;

    h_localTrigger = (int *)malloc((MAXNBUBBLE + 1)*sizeof(int));
    
    if(h_compressedHistogramNow == NULL || 
       h_compressedHistogramPast == NULL ||
       h_iso == NULL || h_localTrigger == NULL){
      fprintf(stderr, 
	      "No Alloc h_compressedHistogramNow/Past h_iso h_localTrigger\n");
      return DE_EXIT_FAILURE;
    }
    
    MY_CUDA_CHECK( MallocCuda((void **) &d_compressedHistogramNow, 
			      MAXNBUBBLE*sizeof(int) ) );
    MY_CUDA_CHECK( MallocCuda((void **) &d_compressedHistogramPast, 
			      MAXNBUBBLE*sizeof(int) ) );
    
    
    first = FALSE;
  }

  TIMER_START;
  
  if(counter & 1U){
    findDropletsSimV4(grid, block, h_nla, 
		      d_rho, d_flag, 
		      d_flagObstacle, d_label,
		      &d_histogram2, d_bubble2,
		      d_xlab2, d_ylab2, counter);

    if(debug == DE_DEBUG){
      char flagName[1024];
      snprintf(flagName, sizeof(flagName), 
	       "newOutFlagTime%d", time);
      printDeviceIntArrayGridToFile(d_flag, h_nla, h_nx, h_ny, 
				    flagName);

      char labelName[1024];
      snprintf(labelName, sizeof(labelName), 
	       "newOutLabelDropletsTime%d", time);

      printDeviceUnsignedArrayGridToFile(d_label, h_nla, 
					 h_nx, h_ny, 
					 labelName);

    }

    MY_CUDA_CHECK( cudaMemcpy(&nBubbles2, d_bubble2,
			      sizeof(int),
			      cudaMemcpyDeviceToHost) );
    
    
    makeDelaunayUnsortedV4(grid, block, h_nla,
			   d_label, d_bubble2,
			   d_flag, 
			   d_links2, d_vertexCount,
			   d_vertexCountSymm, d_vertexMult,
			   d_xlab2, d_ylab2);

    if(debug == DE_DEBUG){
      char labelName[1024];
      snprintf(labelName, sizeof(labelName), 
	       "newOutLabelVoronoiTime%d", time);

      printDeviceUnsignedArrayGridToFile(d_label, h_nla, 
					 h_nx, h_ny, 
					 labelName);
      char multName[1024];
      snprintf(multName, sizeof(multName), 
	       "newOutMultTime%d", time);

      printDeviceIntArrayGridToFile(d_vertexMult, h_nla, h_nx, h_ny, 
				    multName);

    }

    
  }else{
    findDropletsSimV4(grid, block, h_nla, 
		      d_rho, d_flag, 
		      d_flagObstacle, d_label,
		      &d_histogram1, d_bubble1,
		      d_xlab1, d_ylab1, counter);

    if(debug == DE_DEBUG){
      char flagName[1024];
      snprintf(flagName, sizeof(flagName), 
	       "newOutFlagTime%d", time);
      printDeviceIntArrayGridToFile(d_flag, h_nla, h_nx, h_ny, 
				    flagName);

      char labelName[1024];
      snprintf(labelName, sizeof(labelName), 
	       "newOutLabelDropletsTime%d", time);

      printDeviceUnsignedArrayGridToFile(d_label, h_nla, 
					 h_nx, h_ny, 
					 labelName);
    }
    
    MY_CUDA_CHECK( cudaMemcpy(&nBubbles1, d_bubble1,
			      sizeof(int),
			      cudaMemcpyDeviceToHost) );
    
    makeDelaunayUnsortedV4(grid, block, h_nla, 
			   d_label, d_bubble1,
			   d_flag,
			   d_links1, d_vertexCount,
			   d_vertexCountSymm,
			   d_vertexMult,
			   d_xlab1, d_ylab1);

    if(debug == DE_DEBUG){
      char labelName[1024];
      snprintf(labelName, sizeof(labelName), 
	       "newOutLabelVoronoiTime%d", time);

      printDeviceUnsignedArrayGridToFile(d_label, h_nla, 
					 h_nx, h_ny, 
					 labelName);
      char multName[1024];
      snprintf(multName, sizeof(multName), 
	       "newOutMultTime%d", time);

      printDeviceIntArrayGridToFile(d_vertexMult, h_nla, h_nx, h_ny, 
				    multName);

    }
      
  }
  
  if(counter > 0){    
    static int *d_linksNow = NULL, *d_linksPast = NULL;
    static int *d_bubbleNow = NULL, *d_bubblePast = NULL;
    static int *d_histogramNow = NULL, *d_histogramPast = NULL;
    static float *d_xlabNow = NULL, *d_xlabPast = NULL;
    static float *d_ylabNow = NULL, *d_ylabPast = NULL;
    static int nBubblesNow = 0, nBubblesPast = 0;
    static int *d_compressedLinksNow = NULL;
    static int *d_compressedLinksPast = NULL;
    static int *h_compressedLinksNow = NULL;
    static int *h_compressedLinksPast = NULL;
    static int eventsCounter = 0;
    
    static int firstAlloc = TRUE;
    
    if(counter & 1U){

      d_linksNow = d_links2;
      d_linksPast = d_links1;
      d_bubbleNow = d_bubble2;
      d_bubblePast = d_bubble1;
      d_xlabNow = d_xlab2; d_ylabNow = d_ylab2;
      d_xlabPast = d_xlab1; d_ylabPast = d_ylab1;
      d_histogramNow = d_histogram2;
      d_histogramPast = d_histogram1;
      nBubblesNow = nBubbles2;
      nBubblesPast = nBubbles1;
    }else{

      d_linksNow = d_links1;
      d_linksPast = d_links2;
      d_bubbleNow = d_bubble1;
      d_bubblePast = d_bubble2;
      d_xlabNow = d_xlab1; d_ylabNow = d_ylab1;
      d_xlabPast = d_xlab2; d_ylabPast = d_ylab2;
      d_histogramNow = d_histogram1;
      d_histogramPast = d_histogram2;
      nBubblesNow = nBubbles1;
      nBubblesPast = nBubbles2;
    }
    
    fprintf(nBubblesOut, "%d %d %d\n", 
	    time, nBubblesNow, eventsCounter);
    
    MY_CUDA_CHECK( cudaMemset(d_iso, 0, sizeof(int)*
			      (MAXNBUBBLE + 1)) );
    MY_CUDA_CHECK( cudaMemset(d_isoCount, 0, sizeof(int)*
			      (MAXNBUBBLE + 1)) );
    memset(h_isoCount, 0, (MAXNBUBBLE + 1)*sizeof(int));

    FILE *checkIsoFile = NULL;
    if(debug == DE_DEBUG){
      char checkIsoCountInit[1024];
      snprintf(checkIsoCountInit, sizeof(checkIsoCountInit),
	       "checkIsoCountInitTime%09d", time);
      checkIsoFile = fopen(checkIsoCountInit, "w");
      
      for(int i=0; i<MAXNBUBBLE + 1; i++){
	fprintf(checkIsoFile, "%d %d\n", i, h_isoCount[i]);
      }
      fprintf(checkIsoFile, "\n\n");
    }

    device_function_findIsomorphism<<<gridUnique, blockUnique>>>
      (d_bubblePast, d_xlabPast, d_ylabPast, 
       d_bubbleNow, d_xlabNow, d_ylabNow,
       d_iso, d_isoCount);

    MY_CUDA_CHECK( cudaDeviceSynchronize() );

    MY_CUDA_CHECK( cudaMemset(d_chkInvert, 0, sizeof(int)) );

    device_function_checkInvert<<<gridUnique, blockUnique>>>(d_bubbleNow, d_isoCount);

    MY_CUDA_CHECK( cudaDeviceSynchronize() );

    h_chkInvert = 0;
    MY_CUDA_CHECK( cudaMemcpy(&h_chkInvert, d_chkInvert, sizeof(int),
			      cudaMemcpyDeviceToHost) );

    if(debug == DE_DEBUG){
      MY_CUDA_CHECK( cudaMemcpy(h_isoCount, d_isoCount, 
				(MAXNBUBBLE + 1)*sizeof(int),
				cudaMemcpyDeviceToHost) );
      
      for(int i=0; i<MAXNBUBBLE + 1; i++){
	fprintf(checkIsoFile, "%d %d\n", i, h_isoCount[i]);
      }
      fflush(checkIsoFile);
      fclose(checkIsoFile);
    
      
      printIsoCMToFile(d_xlabPast, d_ylabPast,
		       d_xlabNow, d_ylabNow,
		       d_bubblePast, d_bubbleNow,
		       d_iso, h_nla, time);
    }


    MY_CUDA_CHECK( cudaMemset(d_linksSwap, 0x77, sizeof(int)*
			      MAXNBUBBLE*MAXNBUBBLE) );
    
    device_function_translateLinks<<<gridTr, blockTr>>>
      (d_linksPast, d_linksSwap, d_iso);


    
    SegSortKeysFromIndices(d_linksNow, MAXNBUBBLE*MAXNBUBBLE, 
			   d_sortKeys, MAXNBUBBLE, *context);
    MY_CUDA_CHECK( cudaMemset(d_linksCountNow, 0, sizeof(int)*
			      (MAXNBUBBLE + 1)) );
    device_function_myUniqueCount<<<gridUnique, blockUnique>>>
      (d_linksNow, d_linksCountNow);

    
    SegSortKeysFromIndices(d_linksSwap, MAXNBUBBLE*MAXNBUBBLE, 
			   d_sortKeys, MAXNBUBBLE, *context);
    MY_CUDA_CHECK( cudaMemset(d_linksCountPast, 0, sizeof(int)*
			      (MAXNBUBBLE + 1)) );    
    device_function_myUniqueCount<<<gridUnique, blockUnique>>>
      (d_linksSwap, d_linksCountPast);
    
    MY_CUDA_CHECK( cudaMemset(d_trigger, 0, sizeof(int)) );
    
    device_function_compareLinksLocal<<<gridUnique, blockUnique>>>
      (d_linksNow, d_linksSwap, d_iso, d_localTrigger);
    
    h_trigger = 0;
    MY_CUDA_CHECK( cudaMemcpy(&h_trigger, d_trigger, sizeof(int),
			      cudaMemcpyDeviceToHost) );

    if(debug == DE_DEBUG){
      printAdjacencyToFile(d_xlabNow, d_ylabNow, 
			   d_bubbleNow, d_linksNow,
			   h_nla, time);
    }

    TIMER_STOP;
    
    if(debug == DE_DEBUG) printf("TOTAL_TIME: %f\n", TIMER_ELAPSED);
    
    fflush(nBubblesOut);
    maxLinksCountNow = findMaxVertexCountByThrust(d_linksCountNow);
    maxLinksCountPast = findMaxVertexCountByThrust(d_linksCountPast);
    sumLinksNow = sumLinksByThrust(d_linksCountNow);
    sumLinksPast = sumLinksByThrust(d_linksCountPast);
    
//    printf("sumLinksNow: %d, sumLinksPast: %d\n",
//	   sumLinksNow, sumLinksPast);
    
    fprintf(nLinksOut, "%d %d %d\n", 
	    time, sumLinksNow, sumLinksPast);
    
    fflush(nLinksOut);
    d_linksPast = d_linksSwap;

    if(h_chkInvert == 0){
    
      if(maxLinksCountNow > maxLinksCountPast) 
	maxLinksCountAbsNow = maxLinksCountNow;
      else maxLinksCountAbsNow = maxLinksCountPast;
      
      if(!firstAlloc && maxLinksCountAbsNow > linksCompressedSize){
	linksCompressedSize = 
	  maxLinksCountAbsNow + maxLinksCountAbsNow/2;
	MY_CUDA_CHECK( cudaMemcpyToSymbol(d_linksCompressedSize, 
					  &linksCompressedSize, 
					  sizeof(int), 0,
					  cudaMemcpyHostToDevice) );
	
	cudaFree(d_compressedLinksNow);
	cudaFree(d_compressedLinksPast);
	MY_CUDA_CHECK( MallocCuda((void**) &d_compressedLinksNow, 
				  linksCompressedSize*MAXNBUBBLE*
				  sizeof(int) ) );
	MY_CUDA_CHECK( MallocCuda((void**) &d_compressedLinksPast, 
				  linksCompressedSize*MAXNBUBBLE*
				  sizeof(int) ) );
	free(h_compressedLinksNow);
	free(h_compressedLinksPast);
	h_compressedLinksNow = 
	  (int *)malloc(linksCompressedSize*MAXNBUBBLE*sizeof(int));
	h_compressedLinksPast = 
	  (int *)malloc(linksCompressedSize*MAXNBUBBLE*sizeof(int));
	if(h_compressedLinksNow == NULL || 
	   h_compressedLinksPast == NULL){
	  fprintf(stderr, "HorreoAlloc\n");
	  return DE_EXIT_FAILURE;
	}
	
      }
      if(firstAlloc){      
	linksCompressedSize = 
	  maxLinksCountAbsNow + maxLinksCountAbsNow/2;
	MY_CUDA_CHECK( cudaMemcpyToSymbol(d_linksCompressedSize, 
					  &linksCompressedSize, 
					  sizeof(int), 0,
					  cudaMemcpyHostToDevice) );
      
	MY_CUDA_CHECK( MallocCuda((void**) &d_compressedLinksNow, 
				linksCompressedSize*MAXNBUBBLE*
				  sizeof(int) ) );
	MY_CUDA_CHECK( MallocCuda((void**) &d_compressedLinksPast, 
				  linksCompressedSize*MAXNBUBBLE*
				  sizeof(int) ) );
	
	h_compressedLinksNow = 
	  (int *)malloc(linksCompressedSize*MAXNBUBBLE*sizeof(int));
	h_compressedLinksPast = 
	  (int *)malloc(linksCompressedSize*MAXNBUBBLE*sizeof(int));
	
	if(h_compressedLinksNow == NULL || 
	   h_compressedLinksPast == NULL){
	  fprintf(stderr, "HorreoAlloc\n");
	  return DE_EXIT_FAILURE;
	}
	
	firstAlloc = FALSE;
      }
      
      MY_CUDA_CHECK( cudaMemset(d_compressedLinksNow, 0x77, sizeof(int)*
				linksCompressedSize*MAXNBUBBLE ) );
      MY_CUDA_CHECK( cudaMemset(d_compressedLinksPast, 0x77, sizeof(int)*
				linksCompressedSize*MAXNBUBBLE ) );
      
      device_function_copyCompressed<<<gridUnique, blockUnique>>>
	(d_linksNow, d_linksPast, d_compressedLinksNow, 
	 d_compressedLinksPast,   
	 d_xlabNow, d_ylabNow, d_xlabPast, d_ylabPast,
	 d_compressedXlabNow, d_compressedYlabNow,
	 d_compressedXlabPast, d_compressedYlabPast,
	 d_histogramNow, d_histogramPast,
	 d_compressedHistogramNow, d_compressedHistogramPast,
	 d_bubbleNow, d_bubblePast);
      

      MY_CUDA_CHECK( cudaMemcpy(h_compressedLinksNow, 
				d_compressedLinksNow,
				linksCompressedSize*MAXNBUBBLE*
				sizeof(int),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_compressedXlabNow, d_compressedXlabNow,
				MAXNBUBBLE*sizeof(float),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_compressedYlabNow, d_compressedYlabNow,
				MAXNBUBBLE*sizeof(float),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_compressedHistogramNow, 
				d_compressedHistogramNow,
				MAXNBUBBLE*sizeof(int),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_linksCountNow, d_linksCountNow,
				MAXNBUBBLE*sizeof(int),
				cudaMemcpyDeviceToHost) );
      
      char delaunayNow[MAXFILENAME];
      if(h_trigger == 1){
	snprintf(delaunayNow, sizeof(delaunayNow), 
		 "%s/delaunayNowTime%09d", outDirTrigger, time);
      }
      if(h_trigger == 0){
	snprintf(delaunayNow, sizeof(delaunayNow), 
		 "%s/delaunayNowTime%09d", outDirNoTrigger, time);
      }
      
      FILE *outDelaunayNow = fopen(delaunayNow, "wb");
      if(outDelaunayNow == NULL){
	fprintf(stderr, "Error opening file %s\n", delaunayNow);
	return DE_EXIT_FAILURE;
      }
      fwrite(&h_nx, sizeof(int), 1, outDelaunayNow);
      fwrite(&h_ny, sizeof(int), 1, outDelaunayNow);
      fwrite(&nBubblesNow, sizeof(int), 1, outDelaunayNow);      
      fwrite(&linksCompressedSize, sizeof(int), 1, outDelaunayNow);
      fwrite(h_compressedLinksNow, sizeof(int), 
	     (nBubblesNow + 1)*linksCompressedSize, 
	     outDelaunayNow);
      fwrite(h_compressedXlabNow, sizeof(float), (nBubblesNow + 1), 
	     outDelaunayNow);
      fwrite(h_compressedYlabNow, sizeof(float), (nBubblesNow + 1), 
	     outDelaunayNow);
      fwrite(h_compressedHistogramNow, sizeof(int), (nBubblesNow + 1), 
	     outDelaunayNow);
      fwrite(h_linksCountNow, sizeof(int), (nBubblesNow + 1), 
	     outDelaunayNow);	
      
      fflush(outDelaunayNow);
      fclose(outDelaunayNow);
      
      
      MY_CUDA_CHECK( cudaMemcpy(h_compressedLinksPast, 
				d_compressedLinksPast,
				linksCompressedSize*MAXNBUBBLE*
				sizeof(int),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_compressedXlabPast, 
				d_compressedXlabPast,
				MAXNBUBBLE*sizeof(float),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_compressedYlabPast, 
				d_compressedYlabPast,
				MAXNBUBBLE*sizeof(float),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_compressedHistogramPast, 
				d_compressedHistogramPast,
				MAXNBUBBLE*sizeof(int),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_linksCountPast, d_linksCountPast,
				MAXNBUBBLE*sizeof(int),
				cudaMemcpyDeviceToHost) );
      char delaunayPast[MAXFILENAME];
      if(h_trigger == 1){
	snprintf(delaunayPast, sizeof(delaunayPast), 
		 "%s/delaunayPastTime%09d", outDirTrigger, time);
      }
      if(h_trigger == 0){
	snprintf(delaunayPast, sizeof(delaunayPast), 
		 "%s/delaunayPastTime%09d", outDirNoTrigger, time);
      }
      
      FILE *outDelaunayPast = fopen(delaunayPast, "wb");
      if(outDelaunayPast == NULL){
	fprintf(stderr, "Error opening file %s\n", delaunayPast);
	return DE_EXIT_FAILURE;
      }
      fwrite(&h_nx, sizeof(int), 1, outDelaunayPast);
      fwrite(&h_ny, sizeof(int), 1, outDelaunayPast);
      fwrite(&nBubblesPast, sizeof(int), 1, outDelaunayPast);      
      fwrite(&linksCompressedSize, sizeof(int), 1, outDelaunayPast);
      fwrite(h_compressedLinksPast, sizeof(int), 
	     (nBubblesPast + 1)*linksCompressedSize, 
	     outDelaunayPast);
      fwrite(h_compressedXlabPast, sizeof(float), (nBubblesPast + 1), 
	     outDelaunayPast);
      fwrite(h_compressedYlabPast, sizeof(float), (nBubblesPast + 1), 
	     outDelaunayPast);
      fwrite(h_compressedHistogramPast, sizeof(int), (nBubblesPast + 1), 
	     outDelaunayPast);
      fwrite(h_linksCountPast, sizeof(int), (nBubblesPast + 1), 
	     outDelaunayPast);	
      
      fflush(outDelaunayPast);
      fclose(outDelaunayPast);
      
      MY_CUDA_CHECK( cudaMemcpy(h_iso, d_iso,
				(MAXNBUBBLE + 1)*sizeof(int),
				cudaMemcpyDeviceToHost) );
      MY_CUDA_CHECK( cudaMemcpy(h_isoCount, d_isoCount,
				(MAXNBUBBLE + 1)*sizeof(int),
				cudaMemcpyDeviceToHost) );
      
      MY_CUDA_CHECK( cudaMemcpy(h_localTrigger, d_localTrigger,
				(MAXNBUBBLE + 1)*sizeof(int),
				cudaMemcpyDeviceToHost) );
      
      for(int i=1; i<=nBubblesPast; i++){ 
	h_isoM1[h_iso[i]] = i; 
	uniqueLabels[i] = h_iso[uniqueLabels[i]];
      }

      memset(uniqueLabelsM1, 0, (MAXNBUBBLE + 1)*sizeof(int));
      for(int i=1; i<=nBubblesPast; i++) uniqueLabelsM1[uniqueLabels[i]] = i;
      
      if(debug == DE_DEBUG){
	printf("nBubbles: %d\n", nBubblesNow);
	for(int i=0; i<MAXNBUBBLE; i++){
	  if(h_localTrigger[i] == 1){
	    printf("[%d]: ", h_iso[i]);
	    for(int j=0; j<linksCompressedSize; j++){
	      int index = h_iso[i]*linksCompressedSize + j;
	      printf("%09d ", h_compressedLinksNow[index]);
	    }
	    printf("\n");
	    printf("[%d]: ", i);
	    for(int j=0; j<linksCompressedSize; j++){
	      int index = i*linksCompressedSize + j;
	      if(h_compressedLinksPast[index] < MAXNBUBBLE){
		printf("%09d:%09d ", 
		       h_compressedLinksPast[index],
		       h_isoM1[h_compressedLinksPast[index]]);
	      }
	    }
	    printf("\n\n");
	  }
	}
      }

#if 0
      FILE *dropletsFile = fopen("delaunayTriggerDir/dropletsInfo", "a+");
      if(dropletsFile == NULL){
	fprintf(stderr, "Error opening file %s\n", dropletsFile);
	return DE_EXIT_FAILURE;
      }
      
      for(int i=1; i<=nBubblesPast; i++){
	fprintf(dropletsFile,
		"%010d %05d %05d %05d %14.7e %14.7e %14.7e %14.7e "
		"%010d %010d\n",
		time, i, i, h_iso[i],
		h_compressedXlabNow[h_iso[i]],
		h_compressedYlabNow[h_iso[i]],
		h_compressedXlabPast[i],
		h_compressedYlabPast[i],
		h_compressedHistogramNow[h_iso[i]],
		h_compressedHistogramPast[i]);
      }
      fprintf(dropletsFile, "\n");
      fclose(dropletsFile);
#endif
      
      char delaunayIsoTrigger[MAXFILENAME];
      if(h_trigger == 1){
	snprintf(delaunayIsoTrigger, sizeof(delaunayIsoTrigger), 
		 "%s/delaunayIsoTriggerTime%09d", outDirTrigger, time);
      }
      if(h_trigger == 0){
	snprintf(delaunayIsoTrigger, sizeof(delaunayIsoTrigger), 
		 "%s/delaunayIsoTriggerTime%09d", outDirNoTrigger, time);
      }
      FILE *outDelaunayIsoTrigger = fopen(delaunayIsoTrigger, "wb");
      if(outDelaunayIsoTrigger == NULL){
	fprintf(stderr, "Error opening file %s\n", delaunayIsoTrigger);
	return DE_EXIT_FAILURE;
      }
      
      fwrite(h_iso, sizeof(int), (nBubblesPast + 1), 
	     outDelaunayIsoTrigger);
      fwrite(h_isoCount, sizeof(int), (nBubblesNow + 1), 
	     outDelaunayIsoTrigger);	
      fwrite(h_localTrigger, sizeof(int), (nBubblesPast + 1), 
	     outDelaunayIsoTrigger);	
      
      fflush(outDelaunayIsoTrigger);
      fclose(outDelaunayIsoTrigger);
      
      if(h_trigger == 1){	
	int rv = findAndDumpBreakingArisingDV2(h_localTrigger, 
					       h_iso, h_isoCount, h_isoM1,
					       uniqueLabels, uniqueLabelsM1,
					       uniqueCouplesBreaking, 
					       uniqueCouplesArising,
					       uniqueCouplesBreakingTime, 
					       uniqueCouplesArisingTime,
					       h_compressedLinksPast,
					       h_compressedXlabPast,
					       h_compressedYlabPast,
					       h_compressedHistogramPast,
					       h_linksCountPast,
					       h_compressedLinksNow,
					       h_compressedXlabNow,
					       h_compressedYlabNow,
					       h_compressedHistogramNow,
					       h_linksCountNow,
					       linksCompressedSize,
					       linksCompressedSize, time,
					       nBubblesNow, nBubblesPast,
					       outDirTrigger, debug);
	if(rv == DE_EXIT_FAILURE) return DE_EXIT_FAILURE;
	eventsCounter++;            
      }
    
    }else{
      memset(uniqueCouplesBreaking, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
      memset(uniqueCouplesArising, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
      memset(uniqueCouplesBreakingTime, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
      memset(uniqueCouplesArisingTime, 0, (MAXNBUBBLE + 1)*(MAXNBUBBLE + 1)*sizeof(int));
      
      for(int i=0; i<MAXNBUBBLE + 1; i++) uniqueLabels[i] = i;      
    }
    
    if(nBubblesNow != nBubblesPast){
      h_trigger = DE_BUBBLE_CHANGE;
      fflush(nBubblesOut);
    }
  }
  
  if(h_trigger == 0 || h_trigger == 1 || 
     h_trigger == DE_BUBBLE_CHANGE || h_chkInvert == 1) counter++;

  if(h_chkInvert == 1) h_trigger = DE_INVERT;
  
  return h_trigger;
}

extern int findMaxVertexCountByThrust(int *d_linksCount){
  thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_linksCount);
  thrust::device_vector<int> D(dev_ptr, dev_ptr + MAXNBUBBLE + 1);
  return thrust::reduce(D.begin(), D.end(), (int) 0, thrust::maximum<int>());
}

extern int sumLinksByThrust(int *d_linksCount){
  thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_linksCount);
  thrust::device_vector<int> D(dev_ptr, dev_ptr + MAXNBUBBLE + 1);
  return thrust::reduce(D.begin(), D.end(), (int) 0, thrust::plus<int>());
}

extern int *cntlabelbythrust(int h_nla, unsigned int *d_label,
			     int k){
  int *d_histogram;
  static thrust::device_vector<int> histogram;    
  static thrust::device_vector<int> histogram2;    
  thrust::device_ptr<unsigned int> dev_ptr = thrust::device_pointer_cast(d_label);
  thrust::device_vector<unsigned int> t_label(dev_ptr, dev_ptr + h_nla);

  if(k & 1U){
    dense_histogram(t_label, histogram2);
    d_histogram=thrust::raw_pointer_cast(&histogram2[0]);
  }else{
    dense_histogram(t_label, histogram);
    d_histogram=thrust::raw_pointer_cast(&histogram[0]);
  }

  return d_histogram;
}

template <typename Vector1, typename Vector2>
  void dense_histogram(const Vector1& input,
		       Vector2& histogram){

  typedef typename Vector1::value_type ValueType;
  typedef typename Vector2::value_type IndexType;

  if(input.size()<1) { histogram[0]=0; return; }
  thrust::device_vector<ValueType> data(input);

  thrust::sort(data.begin(), data.end());

  IndexType num_bins = data.back() + 1;

  histogram.resize(data.end()-data.begin());

  thrust::counting_iterator<IndexType> search_begin(0);
  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());

  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());
  histogram[num_bins]=0;

}

extern REAL *readDataVTK(char *fileName, int *dimSizes){
  FILE *fileInput = fopen(fileName, "rb");
  if(fileInput == NULL){
    fprintf(stderr, "Opening file %s error at line %d in %s\n",
            fileName, __LINE__, __FILE__);
    return NULL;
  }


  char inputString[1024];

  fscanf(fileInput, "# vtk DataFile Version 2.0\n");
  fscanf(fileInput, "%s", inputString);
  fscanf(fileInput, "%s", inputString);
  fscanf(fileInput, "%s %s", inputString, inputString);

  fscanf(fileInput, "%s %d %d %d",
         inputString,
         dimSizes + X, dimSizes + Y, dimSizes + Z);

  int dummy;
  fscanf(fileInput, "%s %d %d %d",
         inputString, &dummy, &dummy, &dummy);
  fscanf(fileInput, "%s %d %d %d",
         inputString, &dummy, &dummy, &dummy);

  int N;
  fscanf(fileInput, "%s %d",
         inputString, &N);
  REAL *array = (REAL *)malloc(sizeof(REAL)*dimSizes[X]*dimSizes[Y]);
  
  fscanf(fileInput, "%s %s %s %d",
         inputString, inputString, inputString,
         &dummy);

  fscanf(fileInput, "%s %s\n",
         inputString, inputString);

  for(int y=0; y<dimSizes[Y]; y++){
    for(int x=0; x<dimSizes[X]; x++){
      int index = x + y*dimSizes[X];
      uint64_t swapEndian = 0;
      uint64_t littleEndian = 0;
      fread(&swapEndian, sizeof(uint64_t), 1, fileInput);
      littleEndian = be64toh(swapEndian);
      REAL *swapLittleEndian = (REAL *)(&littleEndian);
      array[index] = *swapLittleEndian;
    }
  }

  fclose(fileInput);
  return array;
}

extern REAL *readDataVTKPxy(char *fileName, int *dimSizes){
  FILE *fileInput = fopen(fileName, "r");
  if(fileInput == NULL){
    fprintf(stderr, "Opening file %s error at line %d in %s\n",
            fileName, __LINE__, __FILE__);
    return NULL;
  }


  char inputString[1024];

  fscanf(fileInput, "# vtk DataFile Version 2.0\n");
  fscanf(fileInput, "%s", inputString);
  fscanf(fileInput, "%s", inputString);
  fscanf(fileInput, "%s %s", inputString, inputString);

  fscanf(fileInput, "%s %d %d %d",
         inputString,
         dimSizes + X, dimSizes + Y, dimSizes + Z);

  int dummy;
  fscanf(fileInput, "%s %d %d %d",
         inputString, &dummy, &dummy, &dummy);
  fscanf(fileInput, "%s %d %d %d",
         inputString, &dummy, &dummy, &dummy);

  int N;
  fscanf(fileInput, "%s %d",
         inputString, &N);
  REAL *array = (REAL *)malloc(sizeof(REAL)*dimSizes[X]*dimSizes[Y]);
  
  fscanf(fileInput, "%s %s %s %d",
         inputString, inputString, inputString,
         &dummy);

  fscanf(fileInput, "%s %s",
         inputString, inputString);

  for(int y=0; y<dimSizes[Y]; y++){
    for(int x=0; x<dimSizes[X]; x++){
      int index = x + y*dimSizes[X];
      fscanf(fileInput, "%lf", array + index);
    }
  }

  fclose(fileInput);
  return array;
}

extern int labelling(int h_nla, int *d_spin,
		     unsigned int *d_label, int block, int grid){

    static int first=TRUE;

    unsigned int mem_N                 = sizeof(int)*(h_nla);
    unsigned int mem_1                 = sizeof(int)*(1);

    static int* h_flag=NULL;
    static unsigned int* d_R;
    static int* d_flag;



    if(first) {
      h_flag        = (int*) malloc(mem_1);

      MY_CUDA_CHECK( MallocCuda((void**) &d_R, mem_N   ) );
      MY_CUDA_CHECK( MallocCuda((void**) &d_flag, mem_1) );
      first=FALSE;

     }

    device_function_init_K<<<grid,block>>> (d_spin, d_label);
    
    
    h_flag[0] = 1;
    while(h_flag[0] != 0){
      h_flag[0] = 0;

      cudaMemcpy(d_flag,h_flag,mem_1,cudaMemcpyHostToDevice);
      
      device_function_scanning_K<<<grid,block>>>(d_label, d_flag);
      device_function_analysis_K<<<grid,block>>>(d_label        );
          
      cudaMemcpy(h_flag,d_flag,mem_1,cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
    }
    
    device_function_shift_label<<<grid,block>>>(d_label);
    cudaDeviceSynchronize();

    return 0;
}

__global__ void device_function_set_flagSimObstacle
(REAL* d_spin, int* d_flag, int* d_flagObstacle){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(!d_pbcy && index<nx) {
        d_flag[index]=UW;
        return;
    }
    if(!d_pbcy && index>=(nla-nx)) {
        d_flag[index]=DW;
        return;
    }

    if(!d_pbcx && index%nx == 0) {
        d_flag[index]=LW;
        return;
    }
    if(!d_pbcx && index%nx == nx - 1) {
        d_flag[index]=RW;
        return;
    }

    int y = index/nx;
    int x = index%nx;
    int indexSpin = ny + 3 + y + (ny + 2)*x;
    int flagObstacle = -d_flagObstacle[indexSpin];
    d_flag[index]=
      ((((d_spin[indexSpin]<threshold)?BUBBLE:EMPTY) & 
	flagObstacle) | ((~flagObstacle) & OBS));

    return;
}

__global__ void device_function_set_flagRectified
(REAL* d_spin, int* d_flag){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(!d_pbcy && index<nx) {
        d_flag[index]=UW;
        return;
    }
    if(!d_pbcy && index>=(nla-nx)) {
        d_flag[index]=DW;
        return;
    }

    if(!d_pbcx && index%nx == 0) {
        d_flag[index]=LW;
        return;
    }
    if(!d_pbcx && index%nx == nx - 1) {
        d_flag[index]=RW;
        return;
    }

    d_flag[index]= ((d_spin[index]<threshold)?BUBBLE:EMPTY);

    return;
}

__global__ void device_function_init_K(int* d_spin, unsigned int* d_label)

{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int la;
    __shared__ int spin[THREADS];
    __shared__ unsigned int lattice_info[THREADS];

    spin[threadIdx.x] = d_spin[index];
    lattice_info[threadIdx.x] = 0;

    __syncthreads();

    la = (index+1)%nx + ((int)(index/nx))*nx;

    if( spin[threadIdx.x] == d_spin[la] ){
        lattice_info[threadIdx.x] |=  0x01;
    }

    la = (index+nx)%nla;

    if( spin[threadIdx.x] == d_spin[la] ){
        lattice_info[threadIdx.x] |=  0x02;
    }

    la = ((((index/nx)+1)*nx)%nla)+(index+1)%nx;

    if( spin[threadIdx.x] == d_spin[la] ){
        lattice_info[threadIdx.x] |=  0x04;
    }

    __syncthreads();
    d_label[index] = (index<<3) | lattice_info[threadIdx.x];

    return;
}

__global__ void device_function_scanning_K(unsigned int* d_label,
                int* d_flag)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int la, label_1, label_2, label_3;
    unsigned int t_label;
    __shared__ unsigned int label[THREADS];

    label[threadIdx.x] = d_label[index];
    label_1 = (label[threadIdx.x]>>3);
    label_2 = nla+1;

    __syncthreads();

    if( ( label[threadIdx.x]&1 ) == 1 ){
      la = (index+1)%nx + ((int)(index/nx))*nx;
      label_2 = min( (d_label[la]>>3), label_2);
    }

    if( ( (label[threadIdx.x]>>1)&1 ) == 1 ){
      la = (index+nx)%nla;
      label_2 = min( (d_label[la]>>3), label_2);
    }

    if( ( (label[threadIdx.x]>>2)&1 ) == 1 ){
      la = ((((index/nx)+1)*nx)%nla)+(index+1)%nx;
      label_2 = min( (d_label[la]>>3), label_2);
    }

    la = (index-1+nx)%nx+((int)(index/nx))*nx;
    label[threadIdx.x] = d_label[la];
    if( ( label[threadIdx.x]&1 ) == 1 ){
      label_2 = min( (label[threadIdx.x]>>3), label_2);
    }

    la = (index-nx+nla)%nla;
    label[threadIdx.x] = d_label[la];
    if( ( (label[threadIdx.x]>>1)&1 ) == 1 ){
      label_2 = min( (label[threadIdx.x]>>3), label_2);
    }

    la =  (index-1+nx)%nx+ ((int)(((index-nx+nla)%nla)/nx))*nx;
    label[threadIdx.x] = d_label[la];
    if( ( (label[threadIdx.x]>>2)&1 ) == 1 ){
      label_2 = min( (label[threadIdx.x]>>3), label_2);
    }

    if(label_2 < label_1){
      t_label = d_label[label_1];
      label_3 = (t_label>>3);
      label_3 = min(label_3,label_2);
      d_label[label_1] = (t_label&0x07) | ( label_3<<3 );
      d_flag[0] = 1;
    }

    return;
}

__global__ void device_function_analysis_K(unsigned int* d_label)

{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ref, t_label;
    __shared__ unsigned int label[THREADS];

    label[threadIdx.x] = d_label[index];
    t_label = index;
    ref = (d_label[t_label]>>3);

    while( ref != t_label){
      t_label = ref;
      ref = (d_label[t_label]>>3);
    }

    d_label[index] = (label[threadIdx.x]&0x07) | ( ref<<3 );

    return;
}

__global__ void device_function_shift_label(unsigned int* d_label)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    d_label[index] = d_label[index]>>3;

    return;
}

__global__ void device_function_findPeriodic(unsigned int* d_label,
					     int* d_labcnt){
  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const int x = k%nx, y = k/nx;

  int smx = k - x + SM(x - 1, nx, d_pbcx);
  int smxmy = smx + (SM(y - 1, ny, d_pbcy) - y)*nx;
  int counter = 0;

  counter += (d_label[k] == d_label[smxmy]);
  
  if(k%nx == 0 && counter > 0 && d_pbcx == 1){
    atomicOr(d_labcnt + d_label[k], PERIODICX);
  }
  if(k/nx == 0 && counter > 0 && d_pbcy == 1){
    atomicOr(d_labcnt + d_label[k], PERIODICY);
  }

  return;
}

__global__ void device_function_compute_mc1New(unsigned int* d_label,
					       int* d_labcnt, 
					       float* d_xlab,
					       float* d_ylab) {
  const int index = threadIdx.x + blockIdx.x*blockDim.x;
  int x = index%nx, y = index/nx;
  int condX = (-((d_labcnt[d_label[index]] & PERIODICX) >> 31) & (-(x > hnx)));
  int condY = ((-((d_labcnt[d_label[index]] & PERIODICY) >> 30)) & (-(y > hny)));

  atomicAdd(d_xlab + d_label[index], x*1.f -(nx & condX & (-d_pbcx))*1.f);

  atomicAdd(d_ylab + d_label[index], y*1.f -(ny & condY & (-d_pbcy))*1.f);

  return;
}

__global__ void device_function_compute_mc2New(int* d_labcnt, 
					       int* d_flag,
					       float* d_xlab,
					       float* d_ylab, 
					       int *bubble) {

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    int labcnt = d_labcnt[index];
    
    int cond = -((labcnt & PERIODICX) >> 31);
    labcnt = ((cond & (labcnt ^ PERIODICX)) | 
	      ((~cond) & labcnt));

    cond = -((labcnt & PERIODICY) >> 30);
    labcnt = ((cond & (labcnt ^ PERIODICY)) | 
	      ((~cond) & labcnt));

    int flag=((labcnt>bst) && (d_flag[index]==BUBBLE));
    
    d_xlab[index] = flag*(d_xlab[index]/(labcnt+!flag))+(!flag)*nla*nla*1.f;
    d_ylab[index] = flag*(d_ylab[index]/(labcnt+!flag))+(!flag)*nla*nla*1.f;
    d_labcnt[index] = labcnt;

    if(d_xlab[index] < 0.) d_xlab[index] += nx;
    if(d_ylab[index] < 0.) d_ylab[index] += ny;

    flag && (bubble[1+atomicAdd(bubble,1)]=index);

    return;
}

__global__ void device_function_computeDistancesV4
(unsigned int *grid, int *d_bubble, int *d_flagObstacle,
 float *seedsX, float *seedsY){

  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const float x = k%nx, y = k/nx;

  float distMin = (float)nla;
  distMin *= distMin;
  distMin *= distMin;

  int labelMin = MAXLINK;
  int testWallObs = (d_flagObstacle[k] != OBS 
		     && d_flagObstacle[k] != UW && d_flagObstacle[k] != DW 
		     && d_flagObstacle[k] != LW && d_flagObstacle[k] != RW);

  for(int i=0; i<d_bubble[0] && testWallObs; i++){
    float distX = fabsf(x - seedsX[d_bubble[i + 1]]);
    float distY = fabsf(y - seedsY[d_bubble[i + 1]]);

    distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
    distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));

    float dist = distX*distX + distY*distY;

    if(distMin > dist){
      labelMin = i + 1;
      distMin = dist;
    }
  }

  grid[k] = labelMin;

  return;
}

__global__ void device_function_countVertex(unsigned int *grid,
                                            int *vertexCount){

  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const int x = k%nx, y = k/nx;

  int spx = k - x + SP(x + 1, nx, d_pbcx);
  int smx = k - x + SM(x - 1, nx, d_pbcx);
  int spy = k + (SP(y + 1, ny, d_pbcy) - y)*nx;
  int smy = k + (SM(y - 1, ny, d_pbcy) - y)*nx;

  int spxpy = spx + (SP(y + 1, ny, d_pbcy) - y)*nx;
  int spymx = spy - x + SM(x - 1, nx, d_pbcx);
  int smxmy = smx + (SM(y - 1, ny, d_pbcy) - y)*nx;
  int smypx = smy - x + SP(x + 1, nx, d_pbcx);

  int frameCount = (grid[spx] != grid[spxpy]);
  frameCount += (grid[spxpy] != grid[spy]);
  frameCount += (grid[spy] != grid[spymx]);
  frameCount += (grid[spymx] != grid[smx]);
  frameCount += (grid[smx] != grid[smxmy]);
  frameCount += (grid[smxmy] != grid[smy]);
  frameCount += (grid[smy] != grid[smypx]);
  frameCount += (grid[smypx] != grid[spx]);

  if(frameCount > 2) atomicAdd(vertexCount + grid[k], 1);

  return;
}

__global__ void device_function_linksWriterNew5
(unsigned int *grid, int *links,
 int *vertexCount, int *vertexCountSymm, int *vertexMult, 
 float *seedsX, float *seedsY,
 int *d_bubble){

  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const int x = k%nx, y = k/nx;
  const int refGrid = grid[k];

  if(refGrid < MAXNBUBBLE){
    float distX = 0., distY = 0.;
    
    int spx = k - x + SP(x + 1, nx, d_pbcx);
    int smx = k - x + SM(x - 1, nx, d_pbcx);
    int spy = k + (SP(y + 1, ny, d_pbcy) - y)*nx;
    int smy = k + (SM(y - 1, ny, d_pbcy) - y)*nx;
    
    int spxpy = spx + (SP(y + 1, ny, d_pbcy) - y)*nx;
    int spymx = spy - x + SM(x - 1, nx, d_pbcx);
    int smxmy = smx + (SM(y - 1, ny, d_pbcy) - y)*nx;
    int smypx = smy - x + SP(x + 1, nx, d_pbcx);
    
    int frameCount = (grid[spx] != grid[spxpy] &&
		      grid[spx] < MAXNBUBBLE && 
		      grid[spxpy] < MAXNBUBBLE);

    frameCount += (grid[spxpy] != grid[spy] &&
		   grid[spxpy] < MAXNBUBBLE &&
		   grid[spy] < MAXNBUBBLE);

    frameCount += (grid[spy] != grid[spymx] &&
		   grid[spy] < MAXNBUBBLE &&
		   grid[spymx] < MAXNBUBBLE);

    frameCount += (grid[spymx] != grid[smx] &&
		   grid[spymx] < MAXNBUBBLE &&
		   grid[smx] < MAXNBUBBLE);

    frameCount += (grid[smx] != grid[smxmy] &&
		   grid[smx] < MAXNBUBBLE &&
		   grid[smxmy] < MAXNBUBBLE);

    frameCount += (grid[smxmy] != grid[smy] &&
		   grid[smxmy] < MAXNBUBBLE &&
		   grid[smy] < MAXNBUBBLE);

    frameCount += (grid[smy] != grid[smypx] && 
		   grid[smy] < MAXNBUBBLE &&
		   grid[smypx] < MAXNBUBBLE);

    frameCount += (grid[smypx] != grid[spx] &&
		   grid[smypx] < MAXNBUBBLE && 
		   grid[spx] < MAXNBUBBLE);

    vertexMult[k] = frameCount;

    if(frameCount > 2){      
      if(grid[spx] != refGrid && grid[spx] < MAXNBUBBLE &&
	 grid[spy] != grid[smypx] && grid[smy] != grid[spxpy]){
	links[refGrid*MAXNBUBBLE + grid[spx]] = grid[spx];
	links[grid[spx]*MAXNBUBBLE + refGrid] = refGrid;
      }	

      if(grid[spy] != refGrid && grid[spy] < MAXNBUBBLE &&
	 grid[spx] != grid[spymx] && grid[smx] != grid[spxpy]){
	links[refGrid*MAXNBUBBLE + grid[spy]] = grid[spy];
	links[grid[spy]*MAXNBUBBLE + refGrid] = refGrid;
      }

      if(grid[smx] != refGrid && grid[smx] < MAXNBUBBLE &&
	 grid[spy] != grid[smxmy] && grid[smy] != grid[spymx]){
	links[refGrid*MAXNBUBBLE + grid[smx]] = grid[smx];
	links[grid[smx]*MAXNBUBBLE + refGrid] = refGrid;
      }

      if(grid[smy] != refGrid && grid[smy] < MAXNBUBBLE &&
	 grid[spx] != grid[smxmy] && grid[smx] != grid[smypx]){
	links[refGrid*MAXNBUBBLE + grid[smy]] = grid[smy];
	links[grid[smy]*MAXNBUBBLE + refGrid] = refGrid;
      }

#if 1
      if(grid[spxpy] != refGrid && grid[spxpy] < MAXNBUBBLE 
	 && grid[spy] != grid[spx]
	 && grid[spx] != grid[spymx] && grid[spy] != grid[smypx]
	 && spxpy != spx && spxpy != spy){
	float distSpx = (float)nla;
	distSpx *= distSpx;
	if(grid[spx] != refGrid && grid[spx] < MAXNBUBBLE){
	  distX = fabsf(SP(x + 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[spx]]] - 0.5);
	  distY = fabsf(SP(y + 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[spx]]] - 0.5);      
	  distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	  distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));
	  distSpx = distX*distX + distY*distY;
	}
	
	float distSpy = (float)nla;
	distSpy *= distSpy;
	if(grid[spy] != refGrid && grid[spy] < MAXNBUBBLE){
	  distX = fabsf(SP(x + 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[spy]]] - 0.5);
	  distY = fabsf(SP(y + 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[spy]]] - 0.5);      
	  distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	  distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));
	  distSpy = distX*distX + distY*distY;
	}

	distX = fabsf(SP(x + 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[spxpy]]] - 0.5);
	distY = fabsf(SP(y + 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[spxpy]]] - 0.5);      
	distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));
	float distSpxpy = distX*distX + distY*distY;

	distX = fabsf(SP(x + 1, nx, d_pbcx)*1.f - seedsX[d_bubble[refGrid]] - 0.5);
	distY = fabsf(SP(y + 1, ny, d_pbcy)*1.f - seedsY[d_bubble[refGrid]] - 0.5);      
	distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));
	float distRef = distX*distX + distY*distY;
		  
	int condSpxpy = ((distSpx > distSpxpy) &&
			 (distSpy > distSpxpy));      

	int condRef = ((distSpx > distRef) &&
		       (distSpy > distRef));      
	
	if(condSpxpy == 1 || condRef == 1){
	  links[refGrid*MAXNBUBBLE + grid[spxpy]] = grid[spxpy];
	  links[grid[spxpy]*MAXNBUBBLE + refGrid] = refGrid;
	}		      
      }

      if(grid[smxmy] != refGrid && grid[smxmy] < MAXNBUBBLE 
	 && grid[smx] != grid[smy]
	 && grid[smx] != grid[smypx] && grid[smy] != grid[spymx]
	 && smxmy != smx && smxmy != smy){	  
	float distSmx = (float)nla;
	distSmx *= distSmx;
	if(grid[smx] != refGrid && grid[smx] < MAXNBUBBLE){
	  distX = fabsf(SM(x - 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[smx]]] + 0.5);
	  distY = fabsf(SM(y - 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[smx]]] + 0.5);      
	  distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	  distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));
	  distSmx = distX*distX + distY*distY;
	}
	
	float distSmy = (float)nla;
	distSmy *= distSmy;
	if(grid[smy] != refGrid && grid[smy] < MAXNBUBBLE){
	  distX = fabsf(SM(x - 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[smy]]] + 0.5);
	  distY = fabsf(SM(y - 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[smy]]] + 0.5);      
	  distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	  distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));
	  distSmy = distX*distX + distY*distY;
	}
	
	distX = fabsf(SM(x - 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[smxmy]]] + 0.5);
	distY = fabsf(SM(y - 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[smxmy]]] + 0.5);      
	distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	float distSmxmy = distX*distX + distY*distY;	
	
	distX = fabsf(SM(x - 1, nx, d_pbcx)*1.f - seedsX[d_bubble[refGrid]] + 0.5);
	distY = fabsf(SM(y - 1, ny, d_pbcy)*1.f - seedsY[d_bubble[refGrid]] + 0.5);      
	distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	float distRef = distX*distX + distY*distY;

	int condSmxmy = ((distSmy > distSmxmy) &&
			 (distSmx > distSmxmy));      
	int condRef = ((distSmy > distRef) &&
		       (distSmx > distRef));      
	
	if(condSmxmy == 1 || condRef == 1){
	  links[refGrid*MAXNBUBBLE + grid[smxmy]] = grid[smxmy];
	  links[grid[smxmy]*MAXNBUBBLE + refGrid] = refGrid;
	}	      
      }
#endif
    
      if(grid[spymx] != refGrid && grid[spymx] < MAXNBUBBLE
	 && grid[spy] != grid[smx]
	 && spymx != spy && spymx != smx){	
	float distSmx = (float)nla;
	distSmx *= distSmx;
	if(grid[smx] != refGrid && grid[smx] < MAXNBUBBLE){
	  distX = fabsf(SM(x - 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[smx]]] + 0.5);
	  distY = fabsf(SP(y + 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[smx]]] - 0.5);      
	  distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	  distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	  distSmx = distX*distX + distY*distY;
	}
	
	float distSpy = (float)nla;
	distSpy *= distSpy;
	if(grid[spy] != refGrid && grid[spy] < MAXNBUBBLE){
	  distX = fabsf(SM(x - 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[spy]]] + 0.5);
	  distY = fabsf(SP(y + 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[spy]]] - 0.5);      
	  distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	  distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	  distSpy = distX*distX + distY*distY;
	}
	
	distX = fabsf(SM(x - 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[spymx]]] + 0.5);
	distY = fabsf(SP(y + 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[spymx]]] - 0.5);      
	distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	float distSpymx = distX*distX + distY*distY;

	distX = fabsf(SM(x - 1, nx, d_pbcx)*1.f - seedsX[d_bubble[refGrid]] + 0.5);
	distY = fabsf(SP(y + 1, ny, d_pbcy)*1.f - seedsY[d_bubble[refGrid]] - 0.5);      
	distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	float distRef = distX*distX + distY*distY;
	
	int condSpymx = ((distSpy > distSpymx) && 
			 (distSmx > distSpymx));      
	int condRef = ((distSpy > distRef) && 
		       (distSmx > distRef));      
	
	if(condSpymx == 1 || condRef == 1){
	  links[refGrid*MAXNBUBBLE + grid[spymx]] = grid[spymx];
	  links[grid[spymx]*MAXNBUBBLE + refGrid] = refGrid;
	}	
      }

      if(grid[smypx] != refGrid && grid[smypx] < MAXNBUBBLE
	 && grid[smy] != grid[spx]
	 && smypx != smy && smypx != spx){
	float distSpx = (float)nla;
	distSpx *= distSpx;
	if(grid[spx] != refGrid && grid[spx] < MAXNBUBBLE){
	  distX = fabsf(SP(x + 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[spx]]] - 0.5);
	  distY = fabsf(SM(y - 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[spx]]] + 0.5);      
	  distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	  distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	  distSpx = distX*distX + distY*distY;
	}
	
	float distSmy = (float)nla;
	distSmy *= distSmy;
	if(grid[smy] != refGrid && grid[smy] < MAXNBUBBLE){
	  distX = fabsf(SP(x + 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[smy]]] - 0.5);
	  distY = fabsf(SM(y - 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[smy]]] + 0.5);      
	  distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	  distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	  distSmy = distX*distX + distY*distY;
	}
	
	distX = fabsf(SP(x + 1, nx, d_pbcx)*1.f - seedsX[d_bubble[grid[smypx]]] - 0.5);
	distY = fabsf(SM(y - 1, ny, d_pbcy)*1.f - seedsY[d_bubble[grid[smypx]]] + 0.5);      
	distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	float distSmypx = distX*distX + distY*distY;	

	distX = fabsf(SP(x + 1, nx, d_pbcx)*1.f - seedsX[d_bubble[refGrid]] - 0.5);
	distY = fabsf(SM(y - 1, ny, d_pbcy)*1.f - seedsY[d_bubble[refGrid]] + 0.5);      
	distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
	distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));	
	float distRef = distX*distX + distY*distY;	
	
	int condSmypx = ((distSmy > distSmypx) &&
			 (distSpx > distSmypx));      

	int condRef = ((distSmy > distRef) &&
		       (distSpx > distRef));      
	
	if(condSmypx == 1 || condRef == 1){	  
	  links[refGrid*MAXNBUBBLE + grid[smypx]] = grid[smypx];
	  links[grid[smypx]*MAXNBUBBLE + refGrid] = refGrid;
	}		
      }    

    }    

  }
  
  return;
}

__global__ void device_function_myUniqueCount(int *links, int *vertexCount){
  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const int offset = MAXNBUBBLE*k;

  int j = 0, test = 1; 
  for(int i=0; i<MAXNBUBBLE && test; i++){
    if(links[offset + i] > links[offset + j])
      links[offset + (++j)] = links[offset + i];
    if(links[offset + i] > MAXNBUBBLE) test = 0;
  }

  vertexCount[k] = j;
  
  return;
}

__global__ void device_function_adjacencyMatrix(int *links){
  const int offset = threadIdx.x*d_maxVertexCount;

  for(int i=0; i<d_maxVertexCount; i++){
      for(int j=0; j<d_maxVertexCount - 1 - i; j++){
        if(links[offset + j] > links[offset + j + 1]){
          unsigned int swap = links[offset + j];
          links[offset + j] = links[offset + j + 1];
          links[offset + j + 1] = swap;
        }
      }
  }

  int k = 0;
  for(int i=0; i<d_maxVertexCount - 1; i++){
    if(links[offset + i] > links[offset + k])
      links[offset + (++k)] = links[offset + i];
  }

  return;
}

__global__ void device_function_findIsomorphism
(int *d_bubble, float *seedsX, float *seedsY,
 int *d_bubble1, float *seedsX1, float *seedsY1,
 int *isomorphism, int *isoCount){

  const int k = threadIdx.x + blockIdx.x*blockDim.x;

  if(k < d_bubble[0]){
    const float x = seedsX[d_bubble[k + 1]];
    const float y = seedsY[d_bubble[k + 1]];
      
    float distMin = (float)nla;
    distMin *= distMin;
    distMin *= distMin;
    
    int labelMin = MAXNBUBBLE;
    
    for(int i=0; i<d_bubble1[0]; i++){
      float distX = fabsf(x - seedsX1[d_bubble1[i + 1]]);
      float distY = fabsf(y - seedsY1[d_bubble1[i + 1]]);
      
      distX -= ((-(distX > hnx)) & (nx & (-d_pbcx)));
      distY -= ((-(distY > hny)) & (ny & (-d_pbcy)));
      
      float dist = distX*distX + distY*distY;
      
      if(distMin > dist){
	labelMin = i + 1;
	distMin = dist;
      }
    }

    isomorphism[k + 1] = labelMin;
    atomicAdd(isoCount + labelMin, 1);
  }

  return;
}

__global__ void device_function_checkInvert(int *d_bubble, int *isoCount){
  const int k = threadIdx.x + blockIdx.x*blockDim.x;

  if(k < d_bubble[0]){
    if(isoCount[k + 1] != 1) atomicOr(&chkInvert, 1);
  }

  return;
}


__global__ void device_function_translateLinks(int *d_links,
					       int *d_linksSwap,
					       int *d_iso){

  const int o = (threadIdx.x + blockIdx.x*MAXNBUBBLE);
  __shared__ int sh_d_iso[MAXNBUBBLE];

  for(int i=threadIdx.x; i<MAXNBUBBLE; i += blockDim.x){
    sh_d_iso[i]=d_iso[i];
  }
  __syncthreads();

  for(int k=o; k < o + MAXNBUBBLE; k += blockDim.x) {
    int rdlinks = d_links[k];
    int cond = (rdlinks != MAXLINK);
    int index = rdlinks;
    
    d_linksSwap[k] = (cond * sh_d_iso[cond*index]) + (!cond * MAXLINK);
  }

  return;
}

__global__ void device_function_translateLinksSelf(int *d_links,
						   int *d_iso){

  const int o = (threadIdx.x + blockIdx.x*MAXNBUBBLE);
  __shared__ int sh_d_iso[MAXNBUBBLE];

  for(int i=threadIdx.x; i<MAXNBUBBLE; i += blockDim.x){
    sh_d_iso[i]=d_iso[i];
  }
  __syncthreads();

  for(int k=o; k < o + MAXNBUBBLE; k += blockDim.x) {
    int rdlinks = d_links[k];
    int cond = (rdlinks != MAXLINK);
    int index = rdlinks;
    
    d_links[k] = (cond * sh_d_iso[cond*index]) + (!cond * MAXLINK);
  }

  return;
}

__global__ void device_function_compareLinksLocal(int *d_links,
						  int *d_linksCompare,
						  int *d_iso,
						  int *d_localTrigger){
  
  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  if(k > 0){
    const int offset1 = k*MAXNBUBBLE;
    const int offset2 = d_iso[k]*MAXNBUBBLE;
    
    int test = TRUE;
    unsigned int compare = 0;
    
    for(int i=0; i<MAXNBUBBLE && test; i++){
      int index1 = offset1 + i;
      int index2 = offset2 + i;
      if(d_linksCompare[index1] < MAXNBUBBLE ||
	 d_links[index2] < MAXNBUBBLE){
	compare |= (d_linksCompare[index1] != d_links[index2]);       
      }
      if(d_linksCompare[index1] > MAXNBUBBLE ||
	 d_links[index2] > MAXNBUBBLE) test = FALSE;
      
    }
    
    atomicOr(&trigger, compare);
    d_localTrigger[k] = compare;
  }
  return;
}

__global__ void device_function_copyCompressed(int *d_linksNow, 
					       int *d_linksPast,
					       int *d_compressedLinksNow, 
					       int *d_compressedLinksPast,
					       float *d_xlabNow, 
					       float *d_ylabNow,
					       float *d_xlabPast, 
					       float *d_ylabPast,
					       float *d_compressedXlabNow, 
					       float *d_compressedYlabNow,
					       float *d_compressedXlabPast, 
					       float *d_compressedYlabPast,
					       int *d_histogramNow, 
					       int *d_histogramPast,
					       int *d_compressedHistogramNow, 
					       int *d_compressedHistogramPast,
					       int *d_bubbleNow, int *d_bubblePast){
  
  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const int offset1 = k*MAXNBUBBLE;
  const int offset2 = k*d_linksCompressedSize;
  
  int testNow = TRUE, testPast = TRUE;

  for(int i=0; i<MAXNBUBBLE && (testNow || testPast); i++){

    int index1 = offset1 + i;
    int index2 = offset2 + i;

    if(d_linksNow[index1] < MAXNBUBBLE){
      d_compressedLinksNow[index2] = d_linksNow[index1];
    }else testNow = FALSE;

    if(d_linksPast[index1] < MAXNBUBBLE){
      d_compressedLinksPast[index2] = d_linksPast[index1];
    }else testPast = FALSE;

  }

  d_compressedXlabNow[k] = d_xlabNow[d_bubbleNow[k]];
  d_compressedYlabNow[k] = d_ylabNow[d_bubbleNow[k]];

  d_compressedXlabPast[k] = d_xlabPast[d_bubblePast[k]];
  d_compressedYlabPast[k] = d_ylabPast[d_bubblePast[k]];
  
  d_compressedHistogramNow[k] = d_histogramNow[d_bubbleNow[k]];
  d_compressedHistogramPast[k] = d_histogramPast[d_bubblePast[k]];

  return;
}

__global__ void device_function_modifyDroplets
(REAL *rho1, REAL *rho2, unsigned int *label, int *flag,
 int *histogramNow, float *R0, float *Gamma, int time, REAL jump){
  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const int x = k%nx, y = k/nx;
  const int indexRho = ny + 3 + y + (ny + 2)*x;
  
  int smx = k - x + SM(x - 1, nx, d_pbcx);
  int spy = k + (SP(y + 1, ny, d_pbcy) - y)*nx;
  int smy = k + (SM(y - 1, ny, d_pbcy) - y)*nx;
  int spx = k - x + SP(x + 1, nx, d_pbcx);

  int spymx = spy - x + SM(x - 1, nx, d_pbcx);
  int smxmy = smx + (SM(y - 1, ny, d_pbcy) - y)*nx;
  int smypx = smy - x + SP(x + 1, nx, d_pbcx);
  int spxpy = spx + (SP(y + 1, ny, d_pbcy) - y)*nx;

  int onBoundary = 0;
  int bndSpx = (flag[spx] == EMPTY), bndSpy = (flag[spy] == EMPTY);
  int bndSmx = (flag[smx] == EMPTY), bndSmy = (flag[smy] == EMPTY);
  int bndSpxpy = (flag[spxpy] == EMPTY), bndSpymx = (flag[spymx] == EMPTY);
  int bndSmxmy = (flag[smxmy] == EMPTY), bndSmypx = (flag[smypx] == EMPTY);
  onBoundary += bndSpx; onBoundary += bndSpy;
  onBoundary += bndSmx; onBoundary += bndSmy;
  onBoundary += bndSpxpy; onBoundary += bndSpymx;
  onBoundary += bndSmxmy; onBoundary += bndSmypx;
  
  if(flag[k] == BUBBLE && onBoundary > 0){

#if 0
    double Rho1 = rho1[indexRho], Rho2 = rho2[indexRho];
    double meanD1 = ((rho1[spxRho] - Rho1)*bndSpx + 
		     (rho1[spyRho] - Rho1)*bndSpy +
		     (rho1[smxRho] - Rho1)*bndSmx + 
		     (rho1[smyRho] - Rho1)*bndSmy +
		     (rho1[spxpyRho] - Rho1)*bndSpxpy*0.707106 + 
		     (rho1[spymxRho] - Rho1)*bndSpymx*0.707106 +
		     (rho1[smxmyRho] - Rho1)*bndSmxmy*0.707106 + 
		     (rho1[smypxRho] - Rho1)*bndSmypx*0.707106)/
      (1.*onBoundary);

    double meanD2 = ((Rho2 - rho2[spxRho])*bndSpx + 
		     (Rho2 - rho2[spyRho])*bndSpy +
		     (Rho2 - rho2[smxRho])*bndSmx + 
		     (Rho2 - rho2[smyRho])*bndSmy +
		     (Rho2 - rho2[spxpyRho])*bndSpxpy*0.707106 + 
		     (Rho2 - rho2[spymxRho])*bndSpymx*0.707106 +
		     (Rho2 - rho2[smxmyRho])*bndSmxmy*0.707106 + 
		     (Rho2 - rho2[smypxRho])*bndSmypx*0.707106)/
      (1.*onBoundary);
    double Sign = ((rho1[spxRho] - rho2[spxRho])*bndSpx + 
		   (rho1[spyRho] - rho2[spyRho])*bndSpy +
		   (rho1[smxRho] - rho2[smxRho])*bndSmx + 
		   (rho1[smyRho] - rho2[smyRho])*bndSmy +
		   (rho1[spxpyRho] - rho2[spxpyRho])*bndSpxpy + 
		   (rho1[spymxRho] - rho2[spymxRho])*bndSpymx +
		   (rho1[smxmyRho] - rho2[smxmyRho])*bndSmxmy + 
		   (rho1[smypxRho] - rho2[smypxRho])*bndSmypx)/(1.*onBoundary);

#endif

    double delta = rho2[indexRho]*Gamma[label[k]];

    rho2[indexRho] += delta;
    rho1[indexRho] -= delta;     

  }

  return;
}

 __global__ void device_function_copyCompressedStochastic
(float *d_xlabNow, float *d_ylabNow,
 float *d_xlabPast, float *d_ylabPast,
 float *d_compressedXlabNow, float *d_compressedYlabNow,
 float *d_compressedXlabPast, float *d_compressedYlabPast,
 int *d_histogramNow, int *d_histogramPast,
 int *d_compressedHistogramNow, int *d_compressedHistogramPast,
 int *d_bubbleNow, int *d_bubblePast){

  const int k = threadIdx.x + blockIdx.x*blockDim.x;

  d_compressedXlabNow[k] = d_xlabNow[d_bubbleNow[k]];
  d_compressedYlabNow[k] = d_ylabNow[d_bubbleNow[k]];
  
  d_compressedXlabPast[k] = d_xlabPast[d_bubblePast[k]];
  d_compressedYlabPast[k] = d_ylabPast[d_bubblePast[k]];
  
  d_compressedHistogramNow[k] = d_histogramNow[d_bubbleNow[k]];
  d_compressedHistogramPast[k] = d_histogramPast[d_bubblePast[k]];

  return;
}

__global__ void device_function_compareLinks(int *d_links,
					     int *d_linksCompare,
					     int *d_iso){
  
  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const int offset1 = k*MAXNBUBBLE;
  const int offset2 = d_iso[k]*MAXNBUBBLE;

  int test = TRUE;
  for(int i=0; i<MAXNBUBBLE && test; i++){

    int index1 = offset1 + i;
    int index2 = offset2 + i;

    if(d_linksCompare[index1] < MAXNBUBBLE){
      unsigned int compare = (d_linksCompare[index1] != d_links[index2]);
      atomicOr(&trigger, compare);
    }
    else test = FALSE;
  }
  
  return;
}

__global__ void device_function_cureLinks(int *d_links){
  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  const int offset = k*MAXNBUBBLE;
  
  for(int i=0; i<MAXNBUBBLE; i++){
    int index1 = offset + i;
    
    if(d_links[index1] < MAXNBUBBLE){
      int index2 = (d_links[index1]*MAXNBUBBLE + k);
      d_links[index2] = k;
    }

  }

  return;
}

__global__ void device_function_preparationInit
(float *R0, float *Gamma, unsigned int *label, int *histogram, REAL var){
  const int k = threadIdx.x + blockIdx.x*blockDim.x;
  R0[label[k]] = sqrt(1.0*histogram[label[k]]/3.14159);
  Gamma[label[k]] = var;
  return;
}
