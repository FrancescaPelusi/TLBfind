/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#ifndef _DELAUNAY_CUDA_H
#define _DELAUNAY_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iomanip>
#include <iterator>

#include "moderngpu.cuh"

using namespace mgpu;

#define SP(a, m, p) ((a&(~(-(a >= m)))&(-p)) | ((a - (a >= m))&(~(-p))))
#define SM(a, m, p) (((a+((-(a < 0))&m))&(-p)) | ((a + (a < 0))&(~(-p))))

#define MAXLINK 0x77777777
#define PERIODICX 0x80000000
#define PERIODICY 0x40000000

#define MAXFILENAME 1024
#define MAXDIRNAME 1024
#define MAXCMD 1024

#define REAL double
#define X 0
#define Y 1
#define Z 2
#define EPS 0.0001
#define DIM 2

#define MAXNBUBBLE 2048
#define DE_EXIT_FAILURE -55
#define DE_BUBBLE_CHANGE -724
#define DE_DEGENERATE -777
#define DE_INVERT -111

#define MY_CUDA_CHECK( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

typedef struct {
  int *h_iso, *h_isoCount, *h_localTrigger, *h_isoM1;
}isoTriggerType;

typedef struct {
  int h_nx, h_ny;
  int nBubbles, linksCompressedSize;
  int *h_compressedLinks, *h_compressedHistogram;
  int *h_linksCount;
  float *h_compressedXlab, *h_compressedYlab;
  isoTriggerType *isoTrigger;
}delaunayType;

typedef struct {
  float *vectorX, *vectorY;
  float *pointX, *pointY;
  int N;
}vectorField2D;

typedef struct{
  float *scalar;
  float *pointX, *pointY;
  int N;
}scalarField2D;

typedef struct{
  scalarField2D *field;
  int dimSizes[DIM], *localCounter;
  int d;
}scalarField2DS;

enum {NOW, PAST};
enum {DE_NODEBUG, DE_DEBUG};

extern float scalarField2D_sumElements(scalarField2D *field);

extern float float_normalizeByIntConstant(float x, int norm);

extern float float_normalizeByFloatConstant(float x, float norm);

extern vectorField2D *vectorField2D_minusConstant(vectorField2D *field, float conX,
						  float conY);
extern scalarField2D *scalarField2D_minusConstant(scalarField2D *field, float cost);

extern scalarField2D *vectorField2D_absValue(vectorField2D *field);

extern scalarField2D *vectorField2D_fromVector2DToScalar2D(vectorField2D *field, int xy);

extern int *scalarField2D_computeTempVeloDropSquaredCell2(delaunayType *Now, scalarField2D *temp,scalarField2D *velo,
							  scalarField2D *tempDrop,scalarField2D *veloDrop,
							  scalarField2D *gradTempDrop);

extern int *scalarField2D_computeTempVeloDropSquaredCellSP(scalarField2D *temp,scalarField2D *velo,scalarField2D *tempDrop,
							   scalarField2D *veloDrop,scalarField2D *gradTempDrop,int LX, int LY,
							   int nBubblesX, int nBubblesY, float diameter, float mean,float radius);

extern int scalarField2D_computeAndPrintNusseltNumberDroplets(const char *outName, const char *outNameAve, int time,
							      scalarField2D *tempDrop, scalarField2D *veloDrop,scalarField2D *gradTempDrop,
							      float deltaTemp, int LY);

extern int scalarField2D_countPairs(scalarField2D *field, float r, int xy);

extern void scalarField2DS_rectifyRho(REAL *h_rho, scalarField2DS *rho);

extern float scalarField2DS_findSup(scalarField2DS *sField);

extern float scalarField2D_findSup(scalarField2D *field);

extern int scalarField2DS_dumpToFile(const char *fileName, scalarField2DS *sField);

extern int scalarField2DS_readFromFile(const char *fileName, scalarField2DS *sField);

extern int scalarField2D_readTemperatureFromFile(char *fileName, scalarField2D *field, int LX, int LY);
extern int scalarField2D_readTemperatureProfileFromFile(char *fileName, scalarField2D *field, int LY);

extern int scalarField2D_readVelocityZFromFile(char *fileName, scalarField2D *field, int LX, int LY);

extern void scalarField2DS_coarseGraining(scalarField2DS *sField,
					  scalarField2DS *sFieldCoarse);

extern void scalarField2DS_dimensionalReduction(scalarField2DS *sField,
						scalarField2DS *sFieldRed);

extern int scalarField2DS_powerNfloat(scalarField2DS *A, scalarField2DS *res,
				      double power);

extern int scalarField2D_powerNfloat(scalarField2D *A, scalarField2D *res,
				     double power);

extern void scalarField2DS_setLocalCounter(scalarField2DS *sField, int value);

extern void scalarField2DS_selfNormalize(scalarField2DS *sField);

extern void scalarField2DS_printToStreamPM3D(scalarField2DS *sField,
					     FILE *stream);
extern void scalarField2DS_printToFilePM3D(scalarField2DS *sField,
					   const char *fileName,
					   const char *mode);

extern void scalarField2D_normalizeByConstant(scalarField2D *field,
					      float norm);
extern void scalarField2DS_normalizeByConstant(scalarField2DS *sField,
					       float norm);

extern void scalarField2DS_printToStream(scalarField2DS *sField, FILE *stream);
extern void scalarField2DS_printToFile(scalarField2DS *sField,
				       const char *fileName,
				       const char *mode);

extern void scalarField2D_sumOverScalarField2DS(scalarField2D *uField,
						scalarField2DS *sField);

extern int scalarField2DS_Clean(scalarField2DS *field);
extern scalarField2DS *scalarField2DS_Alloc(int *dimSizes, int d);
extern int scalarField2DS_Free(scalarField2DS *field);

extern int scalarField2D_Clean(scalarField2D *field);
extern scalarField2D *scalarField2D_Alloc(int N);
extern int scalarField2D_Free(scalarField2D *field);

extern int scalarField2D_printToStream(scalarField2D *field, FILE *stream);
extern int scalarField2D_printToStreamPlusTime(scalarField2D *field, FILE *stream, int time);
extern int scalarField2D_printToFile(scalarField2D *field, const char *outFile, const char *mode);
extern int scalarField2D_printToFilePlusTime(scalarField2D *field, const char *outFile, int time);

extern int scalarField2DS_minus(scalarField2DS *A, scalarField2DS *B,
				scalarField2DS *res);

extern int scalarField2D_minus(scalarField2D *A, scalarField2D *B,
			       scalarField2D *res);

extern scalarField2D *scalarField2D_diff(scalarField2D *A, 
					 scalarField2D *B);

extern REAL *readDataVTKPxy(char *fileName, int *dimSizes);
extern REAL *readDataVTK(char *fileName, int *dimSizes);

extern void printRepTrianglesToFileHostNow(delaunayType *delaunay, int time,
					   const char *outDir);
extern int printSupNorm_vectorField2D(vectorField2D *field, int *uniqueLabels,
				      const char *fileName, int time);

extern int findSupNormIndexUnique_vectorField2D
(vectorField2D *field, int *uniqueLabels);

extern int vectorField2D_Clean(vectorField2D *field);
extern vectorField2D *vectorField2D_Alloc(int N);
extern int vectorField2D_Free(vectorField2D *field);

extern int timeVariablePrintToFile(float variable,
				   const char *outFile, int time);

extern vectorField2D *displacementsField(delaunayType *Now, delaunayType *Past,
					 int pbcx, int pbcy);

extern vectorField2D *vectorField2D_relativeDisplacements(vectorField2D *displacements,
							  delaunayType *Now, delaunayType *Past);

extern int vectorField2D_printToFile(vectorField2D *field, const char *outFile);
extern int vectorField2D_printToStream(vectorField2D *field, FILE *stream);
extern int displacementsFieldPrintToFile(vectorField2D *field, const char *outFile, 
					 int time, int *uniqueLabels, delaunayType *Now,
					 delaunayType *Past);
extern int displacementsFieldAveragesPrintToFile(vectorField2D *field, const char *outFile,
						 int *uniqueLabels,delaunayType *Now,int time);

extern void delaunayType_clean(delaunayType *delaunay);
extern void isoTriggerType_clean(isoTriggerType *isoTrigger);

extern void printBubblesToFileHost(delaunayType *delaunay, int time,
				   const char *outDir, int state);

extern void printBubblesToFileHostAppend(delaunayType *delaunay, int time,
					 const char *outDir, int state);

extern void printBubblesToFileHostAreasAveragesAppend(delaunayType *delaunay, int time,
						      const char *outDir);

extern void printBubblesToFileHostUnique(delaunayType *delaunay, int time,
					 const char *outDir, int state,
					 int *uniqueLabels, int chunk);

extern int readTriggerIsoFromFile(char *fileName,
				  delaunayType *Past,
				  delaunayType *Now);

extern void printIsoTriggerToFileHost(delaunayType *Now, delaunayType *Past,
				      int time, const char *outDir);

extern void printAdjacencyToFileHostNow(delaunayType *delaunay, int time,
					const char *outDir);
extern void printAdjacencyToFileHostPast(delaunayType *delaunay, int time,
					 const char *outDir);

extern int readCompressedFromFile(char *fileName, 
				  delaunayType *delaunay);

extern int isOnBoundaryPast(int index, int *h_linksCountPast,
			    int *h_compressedLinksPast, 
			    int linksCompressedSizePast,
			    int *h_isoM1);

extern int isCoupleOnBoundary(int index0, int index1, 			      
			      int *h_compressedLinks, 
			      int linksCompressedSize);

extern int isCoupleNow(int index0, int index1, 			      
		       int *h_compressedLinks, 
		       int linksCompressedSize);

extern int isOnBoundaryNow(int index, int *h_linksCountNow,
			   int *h_compressedLinksNow, 
			   int linksCompressedSizeNow);

extern int findAndDumpBreakingArisingDV2(int *h_localTrigger, 
					 int *h_iso, int *h_isoCount, 
					 int *h_isoM1,
					 int *uniqueLabels, 
					 int *uniqueLabelsM1,
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
					 int linksCompressedSizePast,
					 int linksCompressedSizeNow, int time,
					 int nBubblesNow, int nBubblesPast,
					 const char *outDir, int debug);

extern int findMaxVertexCountByThrust(int *d_linksCount);
extern int sumLinksByThrust(int *d_linksCount);

extern REAL *readBinaryRho(char *fileName, int h_nx, int h_ny);

extern void printIsoCMToFile(float *d_xlab1, float *d_ylab1,
			     float *d_xlab2, float *d_ylab2,
			     int *d_bubble1, int *d_bubble2,
			     int *d_iso, int h_nla, int time);

extern void printAdjacencyToFile(float *d_xlab, float *d_ylab, 
				 int *d_bubble, int d_links,
				 int h_nla, int time);

extern void findDropletsSimV4(int grid, int block, int h_nla, 
			      REAL *d_rho, int *d_flag, 
			      int *d_flagObstacle,
			      unsigned int *d_label,
			      int **d_histogram, int *d_bubble,
			      float *d_xlab, float *d_ylab, 
			      int k);

extern void makeDelaunayUnsortedV4(int grid, int block, int h_nla,
				   unsigned int *d_label, 
				   int *d_bubble,
				   int *d_flagObstacle,
				   int *d_links, 
				   int *d_vertexCount,
				   int *d_vertexCountSymm,
				   int *d_vertexMult,
				   float *d_xlab, float *d_ylab,
				   int debug);


extern int delaunayBinaryDump(int *d_linksNow, int *d_linksPast, 
			      int *d_iso, 
			      int *d_bubbleNow, int *d_bubblePast,
			      int *d_histogramNow, int *d_histogramPast,
			      float *d_xlabNow, float *d_ylabNow,
			      float *d_xlabPast, float *d_ylabPast,
			      int h_nx, int h_ny, int h_nla, 
			      int time, cudaStream_t *stream);

extern int findAndModifyDroplets(REAL *d_rho1, REAL *d_rho2);


extern int delaunayTriggerV4(REAL *d_rho, int *d_flagObstacle, 
			     int xSize, int ySize, 
			     REAL h_threshold, int time, 
			     int whichGPU,
			     cudaStream_t *stream,
			     int pbcx, int pbcy,
			     int debug);

extern int delaunayTriggerVTK(REAL *d_rho, int xSize, int ySize, 
			      REAL h_threshold, int time, int whichGPU,
			      cudaStream_t *stream);

extern void printDeviceIntArrayGridToFile(int *d_array, int size, 
					  int h_nx, int ny, 
					  const char *fileName);

extern void printDeviceIntArrayToFile(int *d_array, int size, 
				      const char *fileName);

extern void printDeviceInt2ArrayToFile(int *d_array1, int *d_array2,
				       int size, 
				       const char *fileName);

extern void printDeviceCMToFile(int *d_array1, 
				float *d_array2, float *d_array3,
				int size1, int size2, int size3,
				const char *fileName);       

extern void printDeviceUnsignedArrayToFile(unsigned int *d_array, 
					   int size, 
					   const char *fileName);

extern void printDeviceUnsignedArrayGridToFile(unsigned int *d_array, 
					       int size, 
					       int h_nx, int ny, 
					       const char *fileName);

extern void printDeviceREALArrayGridToFile(REAL *d_array, int size, 
					   int h_nx, int ny, 
					   const char *fileName);

extern void printDeviceFloatArrayGridToFile(float *d_array, int size, 
					   int h_nx, int h_ny,
					    const char *fileName);

extern void printHostREALArrayGridToFile(REAL *d_array, int size, 
					   int h_nx, int ny, 
					   const char *fileName);
extern int labelling(int h_nla, int *d_spin,
		     unsigned int *d_label, int block, int grid);

extern int *cntlabelbythrust(int h_nla, unsigned int *d_label, int k);

template <typename Vector1, typename Vector2>
  void dense_histogram(const Vector1& input,
		       Vector2& histogram);

__global__ void device_function_checkInvert(int *d_bubble, int *isoCount);

__global__ void device_function_set_flagSimObstacle
(REAL* d_spin, int* d_flag, int* d_flagObstacle);

__global__ void device_function_init_K(int* d_spin, unsigned int* d_label);
__global__ void device_function_scanning_K(unsigned int* d_label,
					   int* d_flag);
__global__ void device_function_analysis_K(unsigned int* d_label);
__global__ void device_function_shift_label(unsigned int* d_label);
__global__ void device_function_findPeriodic(unsigned int* d_label,
					     int* d_labcnt);

__global__ void device_function_compute_mc1New(unsigned int* d_label,
					       int* d_labcnt, 
					       float* d_xlab,
					       float* d_ylab);

__global__ void device_function_compute_mc2New(int* d_labcnt, 
					       int* d_flag,
					       float* d_xlab,
					       float* d_ylab, 
					       int *bubble);

__global__ void device_function_computeDistancesV4
(unsigned int *grid, int *d_bubble, int *d_flagObstacle,
 float *seedsX, float *seedsY);

__global__ void device_function_countVertex(unsigned int *grid,
                                            int *vertexCount);

__global__ void device_function_linksWriterNew5
(unsigned int *grid, int *links,
 int *vertexCount, int *vertexCountSymm, 
 int *vertexMult,
 float *seedsX, float *seedsY,
 int *d_bubble);

__global__ void device_function_myUniqueCount(int *links, int *vertexCount);
__global__ void device_function_adjacencyMatrix(int *links);

__global__ void device_function_findIsomorphism
(int *d_bubble, float *seedsX, float *seedsY,
 int *d_bubble1, float *seedsX1, float *seedsY1,
 int *isomorphism, int *isoCount);

__global__ void device_function_translateLinksSelf(int *d_links,
						   int *d_iso);

__global__ void device_function_translateLinks(int *d_links,
					       int *d_linksCompare,
					       int *d_iso);

__global__ void device_function_compareLinks(int *d_links,
					     int *d_linksCompare,
					     int *d_iso);

__global__ void device_function_compareLinksLocal(int *d_links,
						  int *d_linksCompare,
						  int *d_iso,
						  int *d_localTrigger);

__global__ void device_function_cureLinks(int *d_links);

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
					       int *d_bubbleNow, 
					       int *d_bubblePast);

__global__ void device_function_copyCompressedStochastic
(float *d_xlabNow, float *d_ylabNow,
 float *d_xlabPast, float *d_ylabPast,
 float *d_compressedXlabNow, float *d_compressedYlabNow,
 float *d_compressedXlabPast, float *d_compressedYlabPast,
 int *d_histogramNow, int *d_histogramPast,
 int *d_compressedHistogramNow, int *d_compressedHistogramPast,
 int *d_bubbleNow, int *d_bubblePast);

__global__ void device_function_modifyDroplets
(REAL *rho1, REAL *rho2, unsigned int *label, int *flag,
 int *histogramNow, float *R0, float *Gamma, int time, REAL jump);

__global__ void device_function_preparationInit
(float *R0, float *Gamma, unsigned int *label, int *histogram, REAL var);
#endif
