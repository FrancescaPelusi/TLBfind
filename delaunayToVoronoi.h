/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#ifndef _DELAUNAY_TO_VORONOI_H
#define _DELAUNAY_TO_VORONOI_H

#include <stdio.h>
#include <stdlib.h>
//#include <stdint.h>
#include "delaunayCuda.h"

#define PRINT_ERROR(errorString)\
  {fprintf(stderr,\
	   "error %s: %s at line %d in file %s\n",\
	   __func__, errorString, __LINE__, __FILE__);  \
  }

#define EXIT_ERROR(errorString)\
  {PRINT_ERROR(errorString);\
    return 1;\
  }

#define EXIT_ERROR_NULL(errorString)\
  {PRINT_ERROR(errorString);\
    return NULL;\
  }

#define MASK32TO21BITS 0x001fffff
#define MASK128TO32BITS 0x000000000000000000000000ffffffff
#define MASK128TO64BITS 0x0000000000000000ffffffffffffffff
#define BITSHIFT32 21
#define BITSHIFT64 32
#define BITSHIFT128 64
#define VORONOI_DEBUG 1

enum {FALSE, TRUE};

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATHLEVEL__)
#if GCC_VERSION < 40800
typedef __uint128_t uint128_t;
#else
typedef unsigned __int128 uint128_t;
#endif 
typedef unsigned int pointTag;
typedef long long unsigned int triangleTag;
typedef uint128_t edgeTag;

typedef union {
  edgeTag tag;
  char c[sizeof(edgeTag)];
} edgeTag2BITS;

typedef struct {
  int h_nx, h_ny, pbcx, pbcy;
} domainType;

typedef struct {
  float x, y;
} vectorType;

typedef struct triangle {
  struct triangle *next, *prev;
  vectorType vertices[3], circumCen;
  float circumRad, area;
  triangleTag tag;
  int flag;
} triangleType;

typedef struct generalEdge {
  struct generalEdge *next, *prev;
  vectorType A, B;  
  edgeTag tag;
  int flag;
} edgeType;

typedef struct point {
  // to be chirally ordered to neighbours
  struct point **neighbours;
  // chirally ordered pointers to triangles
  // to be allocated once the number of neighbours is known !!!
  triangleType **triangles, **trianglesVoronoi;
  edgeType **voronoiEdges;
  edgeType **delaunayEdges;
  int neighboursNumber;
  pointTag tag;
  float *x, *y;
} pointType;

typedef struct {
  pointType *array;
  int N;
} pointTypeArray;

// Public Functions
extern void edgeTag_printToStream(edgeTag tag, FILE *stream);
extern void printVoronoiAreasToFileHostAppend(pointTypeArray *points, int time,
					      const char *outDir, int bubble,
					      int bnd, REAL area);

// pointType functions
extern int pointType_Init(pointType *point);
extern pointTypeArray *pointTypeArray_Alloc(int N);
extern int pointType_Free(pointType *point);
extern int pointTypeArray_Free(pointTypeArray *points);
extern pointTypeArray *pointTypeArray_BindToDelaunay(delaunayType *delaunay);
extern void pointType_ChiralOrdering(pointType *point, domainType *domain,
				     int biasFlag, float bias);
extern void pointTypeArray_ChiralOrdering(pointTypeArray *points, 
					  domainType *domain,
					  int biasFlag, float bias);
extern void pointTypeArray_printToStreamCompliant(pointTypeArray *points, 
						  FILE *stream);
extern int pointTypeArray_printToFileCompliant(pointTypeArray *points, 
					       const char *fileName,
					       const char *mode);
extern int pointType_AddNeighbour(pointType *point, pointType *newNeigh);
extern int pointType_restoreSymmetry(pointType *point, int *checkRestore);
extern int pointTypeArray_restoreSymmetry(pointTypeArray *points,
					  int *checkRestore);
extern void pointTypeArray_printToStreamTriangles(pointTypeArray *points, 
						  FILE *stream);
extern int pointTypeArray_naiveTrianglesCount(pointTypeArray *points);
extern int pointType_isCouple(pointType *A, pointType *B);
extern void pointType_K1Swap(pointType **pJ, 
			     pointType **pJP1, pointType **pK1);
extern void pointType_K4Swap(pointType **pJ, 
			     pointType **pJP1, pointType **pK4);
extern int pointType_createVoronoiTriangles(pointType *point, domainType *domain);
extern int pointTypeArray_createVoronoiTriangles(pointTypeArray *points, domainType *domain);

// triangleType functions
extern int triangleType_Init(triangleType *tri);
extern int triangleType_Free(triangleType *tri);
extern int triangleType_ListFree(triangleType *triRoot);
extern triangleType *triangleType_Alloc(void);
extern triangleType *triangleType_AddToList(triangleType *root, triangleType *triangle);
extern triangleType *triangleType_buildTriangle(vectorType *vertices, domainType *domain);

extern triangleTag triangleType_createTagFromPointTag(pointTag a, pointTag b, pointTag c);
extern triangleType *triangleType_buildTriangleFromPoint
(pointType *A, pointType *B, pointType *C, delaunayType *Now, domainType *domain);
extern void triangleType_computeCircumCen(triangleType *tri, domainType *domain);
extern triangleType *triangleType_buildTriangleFromPointArray(pointTypeArray *points, delaunayType *Now, 
							      domainType *domain);
extern int triangleType_listCount(triangleType *triangle);
extern int triangleType_printToStreamTriangles(triangleType *root, 
					       FILE *stream);
extern int triangleType_printToStreamCirc(triangleType *root, FILE *stream);
extern int triangleType_printToFileTriangles(triangleType *root, 
					     const char *fileName,
					     const char *mode);
extern int triangleType_printToFileCirc(triangleType *root,
					const char *fileName,
					const char *mode);
extern int triangleType_lookForTag(triangleType *root, triangleTag tag);
extern int triangleType_printToStreamSingle(triangleType *root, FILE *stream,
					    int extTag);

// Edge Functions
extern int edgeType_Free(edgeType *edge);
extern int edgeType_ListFree(edgeType *edgeRoot);
extern int edgeType_Init(edgeType *edge);
extern edgeType *edgeType_Alloc(void);
extern edgeType *edgeType_AddToList(edgeType *root, edgeType *edge);
extern edgeTag edgeType_createTagFromPointTag(pointTag a, pointTag b);
extern void edgeType_printToStreamEdgesPoint(edgeType *root,
					     FILE *stream);
extern int edgeType_printToFileEdgesPoint(edgeType *root,
					  const char *fileName,
					  const char *mode);
extern edgeType *edgeType_buildEdge(vectorType *A, vectorType *B);
extern edgeType *edgeType_buildEdgeFromPoint(pointType *pA, pointType *pB);
extern edgeType *edgeType_buildEdgeFromPointArray(pointTypeArray *points);
extern int edgeType_listCount(edgeType *edge);
extern int edgeType_lookForTag(edgeType *root, edgeTag tag);
extern edgeType *edgeType_findFlag(edgeType *root, int flagValue);

extern edgeType *edgeType_buildEdgeFromTrianglesCircumCen(triangleType *tA, 
							  triangleType *tB);
extern edgeType *edgeType_buildVoronoiFromPointArray(pointTypeArray *points);
extern void edgeType_computeMidpointPBC(vectorType *middle, edgeType *edge,
					domainType *domain);
extern void edgeType_computeBatchelorExx(edgeType *root, scalarField2D *Exx, 
					 domainType *domain);
extern void edgeType_computeBatchelorEyy(edgeType *root, scalarField2D *Eyy, 
					 domainType *domain);
extern void edgeType_computeBatchelorExy(edgeType *root, scalarField2D *Exy, 
					 domainType *domain);
#endif
