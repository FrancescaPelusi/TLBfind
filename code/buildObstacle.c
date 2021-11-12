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
#include "util.h"
#include "iniparser.h"

#define PI 3.14159265358979323846
#define REAL double

int walldownheight, walllambda, walldownwidth;
int triangle;
int walldown=TRUE, wallup=TRUE;
int nx, ny;
int pbcx, pbcy;
REAL alpha;

void *Malloc(int);
char *Strdup(char *);
void writelog(int, int, const char *, ...); 

static int *flagAlloc(void);
static void flagPrint2File(int *flag);
static void buildUpStepsSymmetric(void);
static void buildUpStepsSymmetricAlpha(void);
static void buildUpStepsSymmetricAlphaLR(void);
static void buildUpStepsSymmetricAlphaTriangle(void);
static void buildFrames(void);

int main(int argc, char **argv){

  char key[MAXSTRLEN];
  char *inputfile=NULL;

  for(int i = 1; i < argc; i++){

    char *po = argv[i];
    if (*po++ == '-') {
      switch (*po++) {
        case 'h':
          SKIPBLANK
          fprintf(stderr,"%s -i inputfile\n", argv[0]);
          exit(OK);
          break;
        case 'i':
          SKIPBLANK
          inputfile=Strdup(po);
          break;
        default:
          fprintf(stderr,"%s -i inputfile\n", argv[0]);
          exit(OK);
          break;
      }
    }
  }

  if(inputfile==NULL) {
    writelog(TRUE,APPLICATION_RC,"no inputfile");
  }

  dictionary *ini = iniparser_load(inputfile);
  if (ini==NULL) {
    writelog(TRUE,APPLICATION_RC,"cannot parse file: %s\n", inputfile);
  }

  READINTFI(nx,"nx");
  READINTFI(pbcx,"periodic boundary condition along x");
  READINTFI(ny,"ny");
  READINTFI(pbcy,"periodic boundary condition along y");
  READINTFI(walldown,"roughWallDown");
  READINTFI(wallup,"roughWallUp");
  READINTFI(walldownheight,"height");
  READINTFI(walllambda,"lambda");
  READINTFI(walldownwidth,"width");
  READREALFI(alpha, "alpha");
  READINTFI(triangle,"asymmetric");

  if(wallup == 0 || walldown == 0) buildFrames();
  if(alpha > 0. && triangle == 0) buildUpStepsSymmetricAlphaLR();
  if(alpha > 0. && triangle == 1) buildUpStepsSymmetricAlphaTriangle();
  else buildUpStepsSymmetric();
  
  return 0;
}

static int *flagAlloc(void){
  int *flag = (int *) Malloc((nx+2)*(ny+2)*sizeof(int));
  for(int i=0;i<=nx+1;i++){
    for(int j=0;j<=ny+1;j++){
      int idx1 = j+(ny+2)*i;
      flag[idx1] = 1;
    }
  }

  return flag;
}

static void flagPrint2File(int *flag){
  FILE* fout=fopen("flag.dat","w");
  for(int i=0;i<=nx+1;i++){
     for(int j=0;j<=ny+1;j++){
       int idx1=j+(ny+2)*i;
       fprintf(fout,"%d %d %d\n",i,j,flag[idx1]);
     }
  }
  fclose(fout);
  return;
}

static void buildFrames(void){
  int *flag = flagAlloc();

  if(pbcx == 0 || pbcy == 1){
    for(int j=0; j<ny + 2; j++){
      int idx0 = j, idx1 = j + (ny + 2)*(nx + 1);
      flag[idx0] = 0;
      flag[idx1] = 0;
    }
  }
  if(pbcx == 1 || pbcy == 0){
    for(int i=0; i<nx + 2; i++){
      int idx0 = 0 + (ny + 2)*i, idx1 = (ny + 1) + (ny + 2)*i;
      flag[idx0] = 0;
      flag[idx1] = 0;
    }
  }
  
  flagPrint2File(flag);
  return;
}

static void buildUpStepsSymmetric(void) {  

  int *flag = flagAlloc();
  
  int contagradini = (walldown || wallup)?(nx)/(walllambda):0;
  if(contagradini){
    int centergradino[contagradini];
    for(int i=0;i<=nx+1;i++){
      int idx1 = 0+(ny+2)*i;
      int idx2 = ny+1+(ny+2)*i;
      flag[idx1] = 0;
      flag[idx2] = 0;
    }    
        
    for(int k=0;k<contagradini;k++){
      centergradino[k] = (int)(0.5*walllambda + k*walllambda);
      
      if(walldown){
	printf("setting flags walldown!\n");
	for(int i=-walldownwidth/2;i<=walldownwidth/2;i++){
	  for(int j=0;j<=walldownheight;j++){
	    int idx1=j+(ny+2)*(centergradino[k]+i);
	    flag[idx1]=0;
	  }
	}
      }
      
      if(wallup){
	printf("setting flags wallup!\n");
	for(int i=-walldownwidth/2;i<=walldownwidth/2;i++){
	  for(int j=ny + 1;j>=ny + 1 - walldownheight;j--){
	    int idx1=j+(ny+2)*(centergradino[k]+i);
	    flag[idx1]=0;	   
	  }
	}
      }
      
    }
  }
  
  flagPrint2File(flag);
  return;
}

static void buildUpStepsSymmetricAlpha(void){  

  int *flag = flagAlloc();
  alpha *= PI;
  
  int contagradini = (walldown || wallup)?(nx)/(walllambda):0;
  if(contagradini){
    int centergradino[contagradini];
    for(int i=0;i<=nx+1;i++){
      int idx1 = 0+(ny+2)*i;
      int idx2 = ny+1+(ny+2)*i;
      flag[idx1] = 0;
      flag[idx2] = 0;
    }
    
    for(int k=0;k<contagradini;k++){
      centergradino[k] = (int)(walllambda - walldownwidth + k*walllambda);
      
      if(walldown){
	printf("setting flags walldown!\n");
	for(int i=-walldownwidth/2;i<=walldownwidth/2;i++){
	 for(int j=0;j<=walldownheight;j++){
	   int idx1=j+(ny+2)*(centergradino[k]+i);
	   if(idx1<0) {
	     printf("Invalid index: %d, check walldownheight\n",idx1);
	     exit(-1);
	   }
	   flag[idx1]=0;
	 }
	}
	int minXAlpha = (int)(walldownheight/tan(alpha));
	printf("tangente alpha=%f minalpha=%d\n", tan(alpha), minXAlpha);
	
	for(int i=0;i<minXAlpha;i++){
	  for(int j=1;j<=((int)(i*tan(alpha)));j++){
	    int x = centergradino[k] - walldownwidth/2 + i - minXAlpha;
	    int idx1=j+(ny+2)*x;
	    if(idx1<0) {
	      printf("Invalid index: %d, check alpha\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	
      }
      
      if(wallup){
	printf("setting flags wallup!\n");
	for(int i=-walldownwidth/2;i<=walldownwidth/2;i++){
	  for(int j=ny + 1;j>=ny + 1 - walldownheight;j--){
	    int idx1=j+(ny+2)*(centergradino[k]+i);
	    if(idx1<0) {
	      printf("Invalid index: %d, check walldownheight\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;	   
	  }
	}
	
	int minXAlpha = (int)(walldownheight/tan(alpha));
	printf("tangente alpha=%f minalpha=%d\n", tan(alpha), minXAlpha);
	
	for(int i=0;i<minXAlpha;i++){
	  for(int j=ny;j>=ny - ((int)(i*tan(alpha)));j--){
	    int x = centergradino[k] - walldownwidth/2 + i - minXAlpha;
	    int idx1=j+(ny+2)*x;
	    if(idx1<0) {
	      printf("Invalid index: %d, check alpha\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	
      }
      
    }
  }
  
  flagPrint2File(flag);
  return;
}

static void buildUpStepsSymmetricAlphaLR(void){  
  
  int *flag = flagAlloc();
  alpha *= PI;
  
  int contagradini = (walldown || wallup)?(nx)/(walllambda):0;
  if(contagradini){
    int centergradino[contagradini];
    for(int i=0;i<=nx+1;i++){
      int idx1 = 0+(ny+2)*i;
      int idx2 = ny+1+(ny+2)*i;
      flag[idx1] = 0;
      flag[idx2] = 0;
    }
    
    for(int k=0;k<contagradini;k++){
      centergradino[k] = (int)(0.5*walllambda + k*walllambda);
      
      if(walldown){
	printf("setting flags walldown!\n");
	for(int i=-walldownwidth/2;i<=walldownwidth/2;i++){
	  for(int j=0;j<=walldownheight;j++){
	    int idx1=j+(ny+2)*(centergradino[k]+i);
	    if(idx1<0) {
	      printf("Invalid index: %d, check walldownheight\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	int minXAlpha = (int)(walldownheight/tan(alpha));
	printf("tangente alpha=%f minalpha=%d\n", tan(alpha), minXAlpha);
	
	for(int i=0;i<minXAlpha;i++){
	  for(int j=1;j<=((int)(i*tan(alpha)));j++){
	    int x = centergradino[k] - walldownwidth/2 + i - minXAlpha;
	    int idx1=j+(ny+2)*x;
	    if(idx1<0) {
	      printf("Invalid index: %d, check alpha\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	
	for(int i=0;i<minXAlpha;i++){
	  for(int j=((int)((minXAlpha - i)*tan(alpha)));j>=1;j--){
	    int x = centergradino[k] + walldownwidth/2 + i;
	    int idx1=j+(ny+2)*x;
	    if(idx1<0) {
	      printf("Invalid index: %d, check alpha\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	
      }
      
      if(wallup){
	printf("setting flags wallup!\n");
	for(int i=-walldownwidth/2;i<=walldownwidth/2;i++){
	  for(int j=ny + 1;j>=ny + 1 - walldownheight;j--){
	    int idx1=j+(ny+2)*(centergradino[k]+i);
	    if(idx1<0) {
	      printf("Invalid index: %d, check walldownheight\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;	   
	  }
	}
	
	int minXAlpha = (int)(walldownheight/tan(alpha));
	printf("tangente alpha=%f minalpha=%d\n", tan(alpha), minXAlpha);
	
	for(int i=0;i<minXAlpha;i++){
	  for(int j=ny;j>=ny - ((int)(i*tan(alpha)));j--){
	    int x = centergradino[k] - walldownwidth/2 + i - minXAlpha;
	    int idx1=j+(ny+2)*x;
	    if(idx1<0) {
	      printf("Invalid index: %d, check alpha\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	
	for(int i=0;i<minXAlpha;i++){
	  for(int j=ny - ((int)((minXAlpha - i)*tan(alpha)));j<=ny;j++){
	    int x = centergradino[k] + walldownwidth/2 + i;
	    int idx1=j+(ny+2)*x;
	    if(idx1<0) {
	      printf("Invalid index: %d, check alpha\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	
	
      }
      
    }
  }
  
  flagPrint2File(flag);
  return;
}

static void buildUpStepsSymmetricAlphaTriangle(void){  
  
  int *flag = flagAlloc();
  alpha *= PI;
  
  int contagradini = (walldown || wallup)?(nx)/(walllambda):0;
  if(contagradini){
    int centergradino[contagradini];
    for(int i=0;i<=nx+1;i++){
      int idx1 = 0+(ny+2)*i;
      int idx2 = ny+1+(ny+2)*i;
      flag[idx1] = 0;
      flag[idx2] = 0;
    }
    
    for(int k=0;k<contagradini;k++){
      centergradino[k] = (int)((0.5+k)*walllambda);
      
      if(walldown){
      	printf("setting flags walldown!\n");

	if(walldownwidth != 0){
	  for(int i=-walldownwidth/2;i<=walldownwidth/2;i++){
	    for(int j=0;j<walldownheight;j++){
	      int idx1=j+(ny+2)*(centergradino[k]+i);
	      if(idx1<0) {
		printf("Invalid index: %d, check walldownheight\n",idx1);
		exit(-1);
	      }
	      flag[idx1]=0;
	    }
	  }
	}
	int minXAlpha = (int)(walldownheight/tan(alpha));
	printf("tangente alpha=%f minalpha=%d\n", tan(alpha), minXAlpha);
	
	for(int i=0;i<minXAlpha;i++){
	  for(int j=1;j<=((int)(i*tan(alpha)));j++){
	    int x = centergradino[k] - walldownwidth/2 + i - minXAlpha;
	    int idx1=j+(ny+2)*x;
	    if(idx1<0) {
	      printf("Invalid index: %d, check alpha\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	
      }
      
      if(wallup){
	printf("setting flags wallup!\n");
	
	if(walldownwidth != 0){
	  for(int i=-walldownwidth/2;i<=walldownwidth/2;i++){
	    for(int j=ny + 1;j>ny + 1 - walldownheight;j--){
	      int idx1=j+(ny+2)*(centergradino[k]+i);
	      if(idx1<0) {
		printf("Invalid index: %d, check walldownheight\n",idx1);
		exit(-1);
	      }
	      flag[idx1]=0;	   
	    }
	  }
	}

	int minXAlpha = (int)(walldownheight/tan(alpha));
	printf("tangente alpha=%f minalpha=%d\n", tan(alpha), minXAlpha);
	
	for(int i=0;i<minXAlpha;i++){
	  for(int j=ny;j>=ny - ((int)(i*tan(alpha)));j--){
	    int x = centergradino[k] - walldownwidth/2 + i - minXAlpha;
	    int idx1=j+(ny+2)*x;
	    if(idx1<0) {
	      printf("Invalid index: %d, check alpha\n",idx1);
	      exit(-1);
	    }
	    flag[idx1]=0;
	  }
	}
	
      }
      
    }
  }
  
  flagPrint2File(flag);
  return;
}
