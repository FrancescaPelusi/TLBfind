/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#define USE_DOUBLE_PRECISION

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <strings.h>
#include <unistd.h>
#if defined(USEMPI)
#include <mpi.h>
#endif
#include <assert.h>
#include "timing.h"
#include "delaunayCuda.h"
#include <endian.h>

#if !defined(USE_DOUBLE_PRECISION)
  #define REAL float
  #define SQRT sqrtf
  #define SIN  sinf
  #define ACOS acosf
  #define EXP  expf
  #define v(value) value##f
#else
  #define REAL double
  #define SQRT sqrt
  #define SIN  sin
  #define ACOS acos
  #define EXP  expf
  #define v(value) value
#endif

#include "util.h"
#include "cuda.h"
#include "cuda_memory.h"
#include "cudamacro.h"
#include "iniparser.h"
#include "dictionary.h"
#if defined(USEMPI)
#include "tlbfind2dmpi.h"
#endif


#define npop 9
#define connections 25
#define cs2 (v(1.0)/v(3.0))
#define cs1 SQRT((cs2))
#define cs22 (v(2.0)*cs2)
#define cssq (v(2.0)/v(9.0))
#define TWOPIT 6.28
#define PIT  3.14
#define DROPELIMINATE 20

void ExchangePsiRho(REAL *, REAL *, REAL *, REAL *, int);
#if defined(USEMPI)
extern void StartMpi(int *, int *, int *, char ***);
struct mpiexch *SetUpMpi(int, int, int);
extern void ExchMpi(struct mpiexch*);
extern void barrier(void);
extern REAL GlobalSum(REAL, int);
extern double MaxTime(double);
extern void StopMpi(void);
void ExchangePsi(REAL *, REAL *, int);
void ExchangeRho(REAL *, REAL *, int);
#endif
void SNAPSHOT_DELAUNAY(REAL *, REAL *, REAL *, REAL *, int ); 

extern "C" void assignDeviceToProcess(int *);
extern "C" FILE *Fopen(const char*, const char*);
extern "C" void *Malloc(int);
extern "C" void Free(void *);
extern "C" char *Strdup(char*);
extern "C" void writelog(int, int, char const*, ...);

enum {CONST,HONEYFMMM,LINEARTEMP,INTERFACE};

int nx, ny;
int nxdp;
int pbcx=TRUE, pbcy=TRUE;
int noutdens, nouttemperature, noutvelo, nouttens, noutave, noutenergy;
int ncheckdelaunay, nmindelaunay, delaunayDebug;
REAL  bubbleThreshold;
int dooutvtk=FALSE, dooutvtkrho1=FALSE, dooutvtkrho2=FALSE, dooutvtktemperature=FALSE, dooutenergy = FALSE;

#define NATB 38
cudaStream_t *stream;
int ngpu=1;
int nprocs=1;
int monogpu=TRUE;
#if defined(USEMPI)
struct mpiexch *interexch;
#endif

long randseed=-1;

DeviceQualConstant HostInt32        sNx;
DeviceQualConstant HostInt32        sNxdp;
DeviceQualConstant HostInt32        sNy;
DeviceQualConstant HostInt32        sNxdp2Ny2;
DeviceQualConstant HostInt32        sNxdpNy;
DeviceQualConstant HostInt32        sNx2;
DeviceQualConstant HostInt32        sNxdp2;
DeviceQualConstant HostInt32        sNy2;
DeviceQualConstant HostInt32        sNxdp1;
DeviceQualConstant HostInt32        sNx1;
DeviceQualConstant HostInt32        sNy1;
DeviceQualConstant HostInt32        sTid;

static unsigned int sBlocksNum;
static unsigned int sThreadsNum;
static unsigned int whichgpu;

DeviceQualConstant REAL             sG1a;
DeviceQualConstant REAL             sG2a;
DeviceQualConstant REAL             sG1r;
DeviceQualConstant REAL             sG2r;
DeviceQualConstant REAL             sG12;
DeviceQualConstant REAL             salphaG;
DeviceQualConstant REAL             sTup;
DeviceQualConstant REAL             sTdown;
DeviceQualConstant REAL             sPERTURB_VELO_AMPLITUDE;

DeviceQualConstant REAL             sRt0, sRt1, sRt2;

DeviceQualConstant REAL             sRhowall1;
DeviceQualConstant REAL             sRhowall2;
DeviceQualConstant REAL             sRho0;

DeviceQualConstant REAL             sRelax1;
DeviceQualConstant REAL             sRelax2;
DeviceQualConstant REAL             sRelaxG;

DeviceQualConstant REAL             sWw0, sWw1, sWw2, sWw3, sWw4, sWw5, sWw6, sWw7, sWw8;

DeviceQualConstant REAL             sP0,  sP1,  sP2,  sP3,  sP4,  sP5,  sP6,  sP7,  sP8,
                                    sP9,  sP10, sP11, sP12, sP13, sP14, sP15, sP16,
                                    sP17, sP18, sP19, sP20, sP21, sP22, sP23, sP24;

DeviceQualConstant REAL             sCx0,  sCx1,  sCx2,  sCx3,  sCx4,  sCx5,  sCx6,  sCx7,  sCx8,
                                    sCx9,  sCx10, sCx11, sCx12, sCx13, sCx14, sCx15, sCx16,
                                    sCx17, sCx18, sCx19, sCx20, sCx21, sCx22, sCx23, sCx24;

DeviceQualConstant REAL             sCy0,  sCy1,  sCy2,  sCy3,  sCy4,  sCy5,  sCy6,  sCy7,  sCy8,
                                    sCy9,  sCy10, sCy11, sCy12, sCy13, sCy14, sCy15, sCy16,
                                    sCy17, sCy18, sCy19, sCy20, sCy21, sCy22, sCy23, sCy24;


DeviceQualConstant REAL             sKappa;

DeviceQualConstant REAL             sUwallup, sUwalldown;

#define declare_i_j   int i, j
#define set_i_j       i = (index / (sNy));           \
                      j = (index - (i*(sNy)))

REAL uwallup;
REAL uwalldown;

REAL rhowall1, rhowall2;
REAL relax1;
REAL relax2;
REAL relaxG;
REAL rhol;
REAL rhog;
REAL rho0;
REAL G1a;
REAL G2a;
REAL G1r;
REAL G2r;
REAL G12;
REAL alphaG;
REAL Tup;
REAL Tdown;
REAL PERTURB_VELO_AMPLITUDE;

REAL spacing;

REAL width;
REAL WD;
REAL threshold_WD;

REAL diameter;

int NUMx,NUMy;

REAL innx, innxny, innxm2;

REAL countflag;

int roughWallDown=TRUE, roughWallUp=TRUE;

#define g12 (G12)
#define sg12 (sG12)

long idum;

typedef struct {
  REAL *p[npop];
} pop_type;

void ExchangePop(pop_type *, pop_type *, int);
void SendPop2CPU(pop_type *, pop_type *, int);
void RecvPopFromCPU(pop_type *, pop_type *, int);

void ExchangePop_thermal(pop_type *, int);
void SendPop2CPU_thermal(pop_type *,  int);
void RecvPopFromCPU_thermal(pop_type *, int);

pop_type f1;
pop_type f2;
pop_type g;

pop_type f1Buf;
pop_type f2Buf;
pop_type gBuf;

REAL cx[connections];
REAL cy[connections];

int idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,idx9;
int idx10,idx11,idx12,idx13,idx14,idx15,idx16,idx17,idx18,idx19;
int idx20,idx21,idx22,idx23,idx24;

enum {AVEROUT, DENSITY, VELOCITY, TENSOR, VTK, LASTOBS};

int icount[LASTOBS], icountconfig=0;

REAL omega;

REAL fpath,visco;

REAL rt0=4./9.,rt1=1./9.,rt2=1./36.;

REAL rhoin1,rhoin2, temperatureIn;
REAL ww[npop],p[connections];

int istep, isteprest=1;
int i,j,k,kk,s,ip,an;

int verbose=FALSE;


void Usage(char *cmd) {
  printf("-----------------------\n");
  printf("TLBfind  \n");
  printf("-----------------------\n");
  printf("Usage: %s -h (this help) -v (verbose) -i inputfile\n", cmd);
}

void adump(REAL * deviceF, char *filename) {
  int i;
  FILE *fout;
  REAL *f;

  f = (REAL*) Malloc((nx+2)*(ny+2)*sizeof(REAL));
  Device_SafeMemoryCopyFromDevice(f, deviceF, REAL, (nx+2)*(ny+2));

  fout = Fopen(filename,"w");
  for(i=0; i<(nx+2)*(ny+2); i++) {
         fprintf(fout,"%g\n",f[i]);
  }
  fclose(fout);

  Free(f);
}

void dump(pop_type deviceF, const char *filename, int taskid) {
  int i;
  FILE *fout;
  pop_type f;
  char scratch[MAXFILENAME];

  for(i=0; i<npop; i++) {
    f.p[i] = (REAL*) Malloc((nxdp+2)*(ny+2)*sizeof(REAL));
    Device_SafeMemoryCopyFromDevice(f.p[i], deviceF.p[i], REAL, (nxdp+2)*(ny+2));
  }
  snprintf(scratch,sizeof(scratch),"%s_%d",filename,taskid);
  fout = Fopen(scratch,"w");
  for(i=0; i<npop; i++) {
     fwrite(f.p[i],sizeof(REAL),(nxdp+2)*(ny+2),fout);
  }
  fclose(fout);

  for(i=0; i<npop; i++) {
   Free(f.p[i]);
  }
  if(taskid==0) {
    fout = Fopen("dumpcount.in","w");
    fprintf(fout,"%d %d", icountconfig, istep);
    for(i=0; i<LASTOBS; i++) {
      fprintf(fout," %d", icount[i]);
    }
    fprintf(fout,"\n");
    fclose(fout);
  }
  icountconfig=icountconfig+1;
}

void restore(pop_type deviceF, const char *filename, int taskid) {
  int i;

  FILE *fin;
  char scratch[MAXFILENAME];
  pop_type f;

  for(i=0; i<npop; i++) {
    f.p[i] = (REAL*) Malloc((nxdp+2)*(ny+2)*sizeof(REAL));
  }

  snprintf(scratch,sizeof(scratch),"%s_%d",filename,taskid);
  fin = Fopen(scratch,"r");
  for(i=0; i<npop; i++) {
    fread(f.p[i],sizeof(REAL),(nxdp+2)*(ny+2),fin);
  }
  fclose(fin);
  fin = Fopen("dumpcount.in","r");
  fscanf(fin,"%d %d", &icountconfig,&isteprest);
  for(i=0; i<LASTOBS; i++) {
    fscanf(fin," %d", &(icount[i]));
  }
  fclose(fin);

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyToDevice(deviceF.p[i], f.p[i], REAL, (nxdp+2)*(ny+2));
    Free(f.p[i]);
  }
}

Device_QualKernel void inithydroDevice(REAL *u1,REAL *v1, REAL *u2,REAL *v2,REAL *utot, REAL *vtot)
{
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2Ny2)
    u1[index]=v(0.0);
    v1[index]=v(0.0);
    u2[index]=v(0.0);
    v2[index]=v(0.0);
    utot[index]=v(0.0);
    vtot[index]=v(0.0);

  Device_ParallelEndRepeat
}

Device_QualKernel void inithydroDevicePerturbation(REAL *u1,REAL *v1, REAL *u2,REAL *v2,REAL *utot, REAL *vtot)
{
  unsigned int idx1;
  
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
      set_i_j; i++; j++;
      idx1=j+(sNy+2)*i;

      u1[idx1]=sPERTURB_VELO_AMPLITUDE*(sNx/TWOPIT)*sin(TWOPIT*i/sNx)*cos(PIT*j/sNy);
      v1[idx1]=-sPERTURB_VELO_AMPLITUDE*(sNy/PIT)*sin(PIT*j/sNy)*cos(TWOPIT*i/sNx);
      u2[idx1]=sPERTURB_VELO_AMPLITUDE*(sNx/TWOPIT)*sin(TWOPIT*i/sNx)*cos(PIT*j/sNy);
      v2[idx1]=-sPERTURB_VELO_AMPLITUDE*(sNy/PIT)*sin(PIT*j/sNy)*cos(TWOPIT*i/sNx);
      
      utot[idx1]=v(0.0);
      vtot[idx1]=v(0.0);
      
      Device_ParallelEndRepeat
}

void inithydro(REAL *u1,REAL *v1,REAL *deviceRho1,REAL *u2,REAL *v2,REAL *utot,REAL *vtot,REAL *deviceRho2, int initcond, int taskid, REAL PERTURB_VELO_AMPLITUDE){

  REAL centerxHC[NUMx * NUMy];
  REAL centeryHC[NUMx * NUMy];
  REAL rholRandom[NUMx * NUMy];

  if(PERTURB_VELO_AMPLITUDE != 0.){
    Device_ExecuteKernel(inithydroDevicePerturbation)(u1, v1, u2, v2, utot, vtot);
  }else{
    Device_ExecuteKernel(inithydroDevice)(u1, v1, u2, v2, utot, vtot);
  }
  
  REAL *rho1 = (REAL*) Malloc((nx+2)*(ny+2)*sizeof(REAL));
  REAL *rho2 = (REAL*) Malloc((nx+2)*(ny+2)*sizeof(REAL));

  int i,j,k,l,x,y;

  FILE *fout;

  rhoin1=v(0.0);
  rhoin2=v(0.0);

  if(randseed>0) {
     srand48(randseed);
  }

  if(taskid==0 || nprocs==1) {

    for(i=0;i<=nx+1;i++){
      for(j=0;j<=ny+1;j++){
        idx1=j+(ny+2)*i;
	rho1[idx1]=v(1.0);
      }
    }

    for(i=0;i<=nx+1;i++){
      for(j=0;j<=ny+1;j++){
        idx1=j+(ny+2)*i;
	rho2[idx1]=v(1.0);
      }
    }

    if(initcond == HONEYFMMM){

      for(j=0;j<NUMy;j++){
	for(i=0;i<NUMx;i++){

	  int myIndex = i + j*NUMx;

	  rholRandom[myIndex]=rhol+WD*2.*(0.5-drand48());

	  if(j%2 == 0){	    
	    centerxHC[myIndex] = spacing + 0.5*diameter + i*(diameter + spacing);
	  }else{
	    centerxHC[myIndex] = 0.5*spacing + i*(diameter + spacing);
	  }
	  
	  centeryHC[myIndex] = spacing + 0.5*diameter + j*0.5*sqrt(3.)*(diameter + spacing);
	}
      }

      for(j=0;j<NUMy;j++){
	for(i=0;i<NUMx;i++){

	  int myIndex = i + j*NUMx;

	  for(k=0;k<NUMy;k++){ 
	    for(l=0;l<NUMx;l++){
	      
	      int checkIndex = l + k*NUMx;
	      
	      double distanceCentersX = centerxHC[myIndex] - centerxHC[checkIndex];
	      double distanceCentersY = centeryHC[myIndex] - centeryHC[checkIndex];

	      if(distanceCentersX > nx*0.5 && pbcx == 1) distanceCentersX -= nx;
	      if(distanceCentersY > ny*0.5 && pbcy == 1) distanceCentersY -= ny;

	      double distanceCenters = sqrt(distanceCentersX*distanceCentersX + distanceCentersY*distanceCentersY);
	      
	      if(distanceCenters <= (diameter + spacing + 1.) && rholRandom[myIndex] >= rhol+WD*threshold_WD && rholRandom[checkIndex] >= rhol+WD*threshold_WD && myIndex != checkIndex){

		rholRandom[myIndex]=rhol;
	      }
	    }
	  }
	}
      }

      for(y=0;y<=ny+1;y++){
        for(x=0;x<=nx+1;x++){
          idx1=y+(ny+2)*x;
	  
          rho1[idx1]=rhol;
          rho2[idx1]=rhog;
	  
          for(i=0;i<NUMx;i++){
	    for(j=0;j<NUMy;j++){

	      int myIndex = i + j*NUMx;
	      
	      double distX = fabs(centerxHC[myIndex] - (double)x);
	      double distY = fabs(centeryHC[myIndex] - (double)y);
	      
	      if(distX > nx*0.5 && pbcx == 1) distX -= nx;
	      if(distY > ny*0.5 && pbcy == 1) distY -= ny;
	      
	      double radius = sqrt(distX*distX + distY*distY);
	      
	      if(radius <= diameter/2.){
		rho1[idx1]=rhog;
		rho2[idx1]=rholRandom[myIndex];
	      }
	    }	    
	  }
	}
      }
    }

    if(initcond == CONST){

      for(i=0;i<=nx+1;i++){
        for(j=0;j<=ny+1;j++){
          idx1=j+(ny+2)*i;
          rho1[idx1]=rhog;
          rho2[idx1]=rhol;
	}
      }

    }

    if(initcond == INTERFACE){

      for(i=0;i<nx/2;i++){
        for(j=0;j<=ny+1;j++){
          idx1=j+(ny+2)*i;
          rho1[idx1]=rhog;
          rho2[idx1]=rhol;
	}
      }

      for(i=nx/2;i<=nx+1;i++){
        for(j=0;j<=ny+1;j++){
          idx1=j+(ny+2)*i;
          rho1[idx1]=rhol;
          rho2[idx1]=rhog;
        }
      }
    }

  }
  Device_Synchronize();
#if defined(USEMPI)  
  if(nprocs>1) {
    if(taskid>0) {
      if(Scatter_recv(rho1+(ny+2), (nxdp)*(ny+2)*sizeof(REAL))<0) {
	writelog(TRUE,APPLICATION_RC,"Error in MPI_Scatter recv (rho1)");
      }
      if(Scatter_recv(rho2+(ny+2), (nxdp)*(ny+2)*sizeof(REAL))<0) {
	writelog(TRUE,APPLICATION_RC,"Error in MPI_Scatter recv (rho2)");
      }
    } else {
	if(Scatter_send(rho1+(ny+2), (nxdp)*(ny+2)*sizeof(REAL))<0) {
          writelog(TRUE,APPLICATION_RC,"Error in MPI_Scatter send (rho1)");
	}
	if(Scatter_send(rho2+(ny+2), (nxdp)*(ny+2)*sizeof(REAL))<0) {
          writelog(TRUE,APPLICATION_RC,"Error in MPI_Scatter send (rho2)");
	}
    }
      Device_SafeMemoryCopyToDevice(deviceRho1+(ny+2), rho1+(ny+2), REAL, (nxdp)*(ny+2));
      Device_SafeMemoryCopyToDevice(deviceRho2+(ny+2), rho2+(ny+2), REAL, (nxdp)*(ny+2));
      ExchangeRho(deviceRho1, deviceRho2, ny+2);
  } else {
#endif	  
    Device_SafeMemoryCopyToDevice(deviceRho1, rho1, REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyToDevice(deviceRho2, rho2, REAL, (nxdp+2)*(ny+2));
#if defined(USEMPI)  
  }
#endif
  
  if(taskid==0||nprocs==1) {
    rhoin1=v(0.0);
    for(i=1;i<=nx;i++){
      for(j=1;j<=ny;j++){
        idx1=j+(ny+2)*i;
        rhoin1+=(REAL)(rho1[idx1]);}
    }
    rhoin1=rhoin1*innxny;
    
    fout=Fopen("init_rho1.dat","w");
    for(i=0;i<=nx+1;i++) {
      for(j=0;j<=ny+1;j++){
        idx1=j+(ny+2)*i;
        fprintf(fout,"%d %d %e \n",i,j,rho1[idx1]);
      }
      fprintf(fout,"\n");
    }
    fclose(fout);
    
    rhoin2=v(0.0);
    for(i=1;i<=nx;i++){
      for(j=1;j<=ny;j++){
        idx1=j+(ny+2)*i;
        rhoin2+=(REAL)(rho2[idx1]);}
    }
    rhoin2=rhoin2*innxny;
    
    fout=Fopen("init_rho2.dat","w");
    for(i=0;i<=nx+1;i++){
      for(j=0;j<=ny+1;j++){
        idx1=j+(ny+2)*i;
        fprintf(fout,"%d %d %e \n",i,j,rho2[idx1]);
      }
      fprintf(fout,"\n");
    }
    fclose(fout);
  }
  Free(rho1);
  Free(rho2);
  
  }

void initVelocityPerturbationRestart_thermal(REAL *u1,REAL *v1,REAL *u2,REAL *v2,REAL *utot,REAL *vtot,REAL PERTURB_VELO_AMPLITUDE){

  printf("Perturbation amplitude = %e\n",PERTURB_VELO_AMPLITUDE);
  if(PERTURB_VELO_AMPLITUDE != 0.){
    Device_ExecuteKernel(inithydroDevicePerturbation)(u1, v1, u2, v2, utot, vtot);
  }else{
    Device_ExecuteKernel(inithydroDevice)(u1, v1, u2, v2, utot, vtot);
  }

}

void inithydro_thermal(REAL *deviceTemp, int initcond, pop_type *pBndUp,pop_type *pBndDown, REAL Tup, REAL Tdown, int taskid){

  REAL *temperature = (REAL*) Malloc((nx+2)*(ny+2)*sizeof(REAL));

  int i,j;

  FILE *fout;

  temperatureIn=v(0.0);

  if(randseed>0) {
     srand48(randseed);
  }

  if(taskid==0 || nprocs==1) {

    if(initcond == CONST){
      for(i=0;i<=nx+1;i++){
	for(j=0;j<=ny+1;j++){
	  idx1=j+(ny+2)*i;
	  temperature[idx1]=Tup; 
	}
      }
    }
      
    if(initcond == LINEARTEMP){
	for(i=0;i<=nx+1;i++){
	  for(j=0;j<=ny+1;j++){
	    idx1=j+(ny+2)*i;
	    temperature[idx1]=Tdown + (j*(Tup - Tdown)/(ny+1.0)); 
	  }
	}
    }

    if(initcond != CONST && initcond != LINEARTEMP){
      printf("Unknown initial condition %d for temperature!!\n",initcond);
    }
  }

  Device_Synchronize();
#if defined(USEMPI)  
  if(nprocs>1) {
    if(taskid>0) {
      if(Scatter_recv(temperature+(ny+2), (nxdp)*(ny+2)*sizeof(REAL))<0) {
	writelog(TRUE,APPLICATION_RC,"Error in MPI_Scatter recv (temperature)");
      }
    } else {
	if(Scatter_send(temperature+(ny+2), (nxdp)*(ny+2)*sizeof(REAL))<0) {
          writelog(TRUE,APPLICATION_RC,"Error in MPI_Scatter send (temperature)");
	}
    }
      Device_SafeMemoryCopyToDevice(deviceTemp+(ny+2), temperature+(ny+2), REAL, (nxdp)*(ny+2));
  } else {
#endif	  
    Device_SafeMemoryCopyToDevice(deviceTemp, temperature, REAL, (nxdp+2)*(ny+2));
#if defined(USEMPI)   
  }
#endif
  
  if(taskid==0||nprocs==1) {
    temperatureIn=v(0.0);
    for(i=1;i<=nx;i++){
      for(j=1;j<=ny;j++){
        idx1=j+(ny+2)*i;
        temperatureIn+=(REAL)(temperature[idx1]);}
    }
    temperatureIn=temperatureIn*innxny;
    
    fout=Fopen("init_temperature.dat","w");
    for(i=0;i<=nx+1;i++) {
      for(j=0;j<=ny+1;j++){
        idx1=j+(ny+2)*i;
        fprintf(fout,"%d %d %e \n",i,j,temperature[idx1]);
      }
      fprintf(fout,"\n");
    }
    fclose(fout);
    
  }
  Free(temperature);
  
  }

Device_QualKernel void initpop_1(pop_type *pF,pop_type *pFeq){
  int ip;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2Ny2)
  for(ip=0;ip<npop;ip++){
    pF->p[ip][index]=pFeq->p[ip][index];
  }
  Device_ParallelEndRepeat

}

void initpop(pop_type *pF,pop_type *pFeq){
  Device_ExecuteKernel(initpop_1)(pF, pFeq);
    Device_Synchronize();
}

Device_QualKernel void setrhobc1_1(REAL *rho,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  REAL massapost;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=0+(sNy2)*(index+1);
    idx2=1+(sNy2)*(index+1);

    massapost=pF->p[0][idx2]+(pMemorydown->p[4][index+1]+pMemorydown->p[7][index+1]+pMemorydown->p[8][index+1])+pF->p[1][idx2]+pF->p[3][idx2]+pF->p[7][idx2]+pF->p[8][idx2]+pF->p[4][idx2];

    rho[idx2]=massapost; 
    rho[idx1]=sRhowall1;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc1obstacle_3(REAL *rho){
  unsigned int idx1;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=0+(sNy2)*(index+1);

    rho[idx1] = sRhowall1;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc1obstacle_4(REAL *rho, int *flag) {
  unsigned int idx1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
      set_i_j; i++; j++;
      idx1=j+(sNy+2)*i;
      rho[idx1]=flag[idx1]?rho[idx1]:sRhowall1;
    }
  }
}


Device_QualKernel void setrhopsibc1_1(REAL *rho,REAL *psi,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  REAL massapost;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=0+(sNy2)*(index+1);
    idx2=1+(sNy2)*(index+1);

    massapost=pF->p[0][idx2]+(pMemorydown->p[4][index+1]+pMemorydown->p[7][index+1]+pMemorydown->p[8][index+1])+pF->p[1][idx2]+pF->p[3][idx2]+pF->p[7][idx2]+pF->p[8][idx2]+pF->p[4][idx2];

    rho[idx2]=massapost;
    rho[idx1]=sRhowall1;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc1_2(REAL *rho,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  REAL massapost;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=sNy1+(sNy2)*(index+1);
    idx2=sNy+(sNy2)*(index+1);

    massapost=pF->p[0][idx2]+(pMemoryup->p[2][index+1]+pMemoryup->p[5][index+1]+pMemoryup->p[6][index+1])+pF->p[1][idx2]+pF->p[3][idx2]+pF->p[5][idx2]+pF->p[6][idx2]+pF->p[2][idx2];

    rho[idx2]=massapost;
    rho[idx1]=sRhowall1;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhopsibc1_2(REAL *rho,REAL *psi,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  REAL massapost;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=sNy1+(sNy2)*(index+1);
    idx2=sNy+(sNy2)*(index+1);

    massapost=pF->p[0][idx2]+(pMemorydown->p[4][index+1]+pMemorydown->p[7][index+1]+pMemorydown->p[8][index+1])+pF->p[1][idx2]+pF->p[3][idx2]+pF->p[7][idx2]+pF->p[8][idx2]+pF->p[4][idx2];

    rho[idx2]=massapost;
    rho[idx1]=sRhowall1;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc1_3(REAL *rho){
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNy2)
    idx1=index+(sNy2)*(sNxdp1);
    idx2=index+(sNy2)*1;

    rho[idx1]=rho[idx2];
  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc1_4(REAL *rho){
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNy2)
    idx1=index+(sNy2)*0;
    idx2=index+(sNy2)*(sNxdp);

    rho[idx1]=rho[idx2];
  Device_ParallelEndRepeat
}

void setrhobc1(REAL *rho,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  Device_ExecuteKernel(setrhobc1_1)(rho, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  Device_ExecuteKernel(setrhobc1_2)(rho, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  if(monogpu) {
    Device_ExecuteKernel(setrhobc1_3)(rho);
        Device_Synchronize();
    Device_ExecuteKernel(setrhobc1_4)(rho);
        Device_Synchronize();
  }
}

void setrhobc1obstacle(REAL *rho,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF, int *flag){
  Device_ExecuteKernel(setrhobc1_1)(rho, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  Device_ExecuteKernel(setrhobc1_2)(rho, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  if(monogpu) {
    Device_ExecuteKernel(setrhobc1obstacle_3)(rho);
        Device_Synchronize();
    Device_ExecuteKernel(setrhobc1obstacle_4)(rho, flag);
        Device_Synchronize();
  }
}


void setrhopsibc1(REAL *rho,REAL *psi,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  Device_ExecuteKernel(setrhopsibc1_1)(rho, psi, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
    Device_ExecuteKernel(setrhopsibc1_2)(rho, psi, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  if(monogpu) {
    Device_ExecuteKernel(setrhobc1_3)(rho);
        Device_Synchronize();
    Device_ExecuteKernel(setrhobc1_4)(rho);
        Device_Synchronize();
  }
}

Device_QualKernel void setrhobc2_1(REAL *rho,pop_type *Memoryup,pop_type *pMemorydown,pop_type *pF){
  REAL massapost;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=0+(sNy2)*(index+1);
    idx2=1+(sNy2)*(index+1);

    massapost=pF->p[0][idx2]+(pMemorydown->p[4][index+1]+pMemorydown->p[7][index+1]+pMemorydown->p[8][index+1])+pF->p[1][idx2]+pF->p[3][idx2]+pF->p[7][idx2]+pF->p[8][idx2]+pF->p[4][idx2];

    rho[idx2]=massapost;
    rho[idx1]=sRhowall2;


  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc2obstacle_3(REAL *rho){

  unsigned int idx1;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=0+(sNy2)*(index+1);

    rho[idx1] = sRhowall2;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc2obstacle_4(REAL *rho, int *flag) {
  unsigned int idx1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
      set_i_j; i++; j++;
      idx1=j+(sNy+2)*i;
      rho[idx1]=flag[idx1]?rho[idx1]:sRhowall2;
    }
  }
}


Device_QualKernel void setrhopsibc2_1(REAL *rho,REAL *psi,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  REAL massapost;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=0+(sNy2)*(index+1);
    idx2=1+(sNy2)*(index+1);

    massapost=pF->p[0][idx2]+(pMemorydown->p[4][index+1]+pMemorydown->p[7][index+1]+pMemorydown->p[8][index+1])+pF->p[1][idx2]+pF->p[3][idx2]+pF->p[7][idx2]+pF->p[8][idx2]+pF->p[4][idx2];

    rho[idx2]=massapost;
    rho[idx1]=sRhowall2;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc2_2(REAL *rho,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  REAL massapost;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=sNy1+(sNy2)*(index+1);
    idx2=sNy+(sNy2)*(index+1);

    massapost=pF->p[0][idx2]+(pMemoryup->p[2][index+1]+pMemoryup->p[5][index+1]+pMemoryup->p[6][index+1])+pF->p[1][idx2]+pF->p[3][idx2]+pF->p[5][idx2]+pF->p[6][idx2]+pF->p[2][idx2];

    rho[idx2]=massapost;
    rho[idx1]=sRhowall2;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhopsibc2_2(REAL *rho,REAL *psi,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  REAL massapost;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)
    idx1=sNy1+(sNy2)*(index+1);
    idx2=sNy+(sNy2)*(index+1);

    massapost=pF->p[0][idx2]+(pMemorydown->p[4][index+1]+pMemorydown->p[7][index+1]+pMemorydown->p[8][index+1])+pF->p[1][idx2]+pF->p[3][idx2]+pF->p[7][idx2]+pF->p[8][idx2]+pF->p[4][idx2];

    rho[idx2]=massapost;
    rho[idx1]=sRhowall2;

  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc2_3(REAL *rho){
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNy2)
    idx1=index+(sNy2)*(sNxdp1);
    idx2=index+(sNy2)*1;
    rho[idx1]=rho[idx2];
  Device_ParallelEndRepeat
}

Device_QualKernel void setrhobc2_4(REAL *rho){
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNy2)
    idx1=index+(sNy2)*0;
    idx2=index+(sNy2)*(sNxdp);
    rho[idx1]=rho[idx2];
  Device_ParallelEndRepeat
}

void setrhobc2(REAL *rho,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  Device_ExecuteKernel(setrhobc2_1)(rho, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  Device_ExecuteKernel(setrhobc2_2)(rho, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  if(monogpu) {
    Device_ExecuteKernel(setrhobc2_3)(rho);
      Device_Synchronize();
    Device_ExecuteKernel(setrhobc2_4)(rho);
      Device_Synchronize();
  }
}

void setrhobc2obstacle(REAL *rho,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF, int *flag){
  Device_ExecuteKernel(setrhobc2_1)(rho, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  Device_ExecuteKernel(setrhobc2_2)(rho, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  if(monogpu) {
    Device_ExecuteKernel(setrhobc2obstacle_3)(rho);
      Device_Synchronize();
    Device_ExecuteKernel(setrhobc2obstacle_4)(rho,flag);
      Device_Synchronize();
  }
}


void setrhopsibc2(REAL *rho,REAL *psi,pop_type *pMemoryup,pop_type *pMemorydown,pop_type *pF){
  Device_ExecuteKernel(setrhopsibc2_1)(rho, psi, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
    Device_ExecuteKernel(setrhopsibc2_2)(rho, psi, pMemoryup, pMemorydown, pF);
    Device_Synchronize();
  if(monogpu) {
    Device_ExecuteKernel(setrhobc2_3)(rho);
      Device_Synchronize();
    Device_ExecuteKernel(setrhobc2_4)(rho);
      Device_Synchronize();
  }
}

void rhocomp(REAL *rho,pop_type *pF);
void forcingconstructWW(REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *psi1self,REAL *psi2self, REAL *utot, REAL *vtot, REAL *tau);
void forcingconstructPBC(REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *psi1self,REAL *psi2self, REAL *tau);

Device_QualKernel void moveplusforcingconstructWW_1(pop_type *pF1,pop_type *pF2,pop_type *pMemoryup1,pop_type *pMemoryup2,pop_type *pMemorydown1,pop_type *pMemorydown2){
  unsigned int ip;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)
    for(ip=0;ip<npop;ip++){
      idx1=sNy+(sNy2)*index;
      idx2=1+(sNy2)*index;

      pMemoryup1->p[ip][index]=pF1->p[ip][idx1];
      pMemoryup2->p[ip][index]=pF2->p[ip][idx1];

      pMemorydown1->p[ip][index]=pF1->p[ip][idx2];
      pMemorydown2->p[ip][index]=pF2->p[ip][idx2];
    }
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructWW_1_thermal(pop_type *pF1, pop_type *pBndUpG, pop_type *pBndDownG, REAL Tup, REAL Tdown, int initcond){
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)
    idx1=sNy+(sNy2)*index;
    idx2=1+(sNy2)*index;

    pF1->p[0][idx2]=(initcond==CONST?Tup:Tdown)*sRt0;
    pF1->p[1][idx2]=(initcond==CONST?Tup:Tdown)*sRt1;
    pF1->p[2][idx2]=(initcond==CONST?Tup:Tdown)*sRt1;
    pF1->p[3][idx2]=(initcond==CONST?Tup:Tdown)*sRt1;
    pF1->p[4][idx2]=(initcond==CONST?Tup:Tdown)*sRt1;
    pF1->p[5][idx2]=(initcond==CONST?Tup:Tdown)*sRt2;
    pF1->p[6][idx2]=(initcond==CONST?Tup:Tdown)*sRt2;
    pF1->p[7][idx2]=(initcond==CONST?Tup:Tdown)*sRt2;
    pF1->p[8][idx2]=(initcond==CONST?Tup:Tdown)*sRt2;

    pF1->p[0][idx1]=Tup*sRt0;
    pF1->p[1][idx1]=Tup*sRt1;
    pF1->p[2][idx1]=Tup*sRt1;
    pF1->p[3][idx1]=Tup*sRt1;
    pF1->p[4][idx1]=Tup*sRt1;
    pF1->p[5][idx1]=Tup*sRt2;
    pF1->p[6][idx1]=Tup*sRt2;
    pF1->p[7][idx1]=Tup*sRt2;
    pF1->p[8][idx1]=Tup*sRt2;

  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructobstacle_1(pop_type *pF1,pop_type *pF2,pop_type *pMemoryup1,pop_type *pMemoryup2,pop_type *pMemorydown1,pop_type *pMemorydown2){
  unsigned int ip;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)
    for(ip=0;ip<npop;ip++){
      idx1=sNy+(sNy2)*index;
      idx2=1+(sNy2)*index;

      pMemoryup1->p[ip][index]=pF1->p[ip][idx1];
      pMemoryup2->p[ip][index]=pF2->p[ip][idx1];


      pMemorydown1->p[ip][index]=pF1->p[ip][idx2];
      pMemorydown2->p[ip][index]=pF2->p[ip][idx2];

    }
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructobstacle_1_thermal(pop_type *pF1,pop_type *pBndUpG,pop_type *pBndDownG){
  unsigned int ip;
  unsigned int idx1, idx2;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)
    for(ip=0;ip<npop;ip++){
      idx1=sNy+(sNy2)*index;
      idx2=1+(sNy2)*index;

      pF1->p[ip][idx1]=pBndUpG->p[ip][index];
      pF1->p[ip][idx2]=pBndDownG->p[ip][index];
    }
  Device_ParallelEndRepeat
}


Device_QualKernel void moveplusforcingconstructWW_2(pop_type *pF1,pop_type *pF2,pop_type *pF1Dest,pop_type *pF2Dest){
  unsigned int idx1, idx2;
  Device_ParallelRepeatUntilIndexIsLessThan(sNy2)
    
    idx1=index+(sNy2)*(sNxdp1);
    idx2=index+(sNy2)*(1);

    pF1Dest->p[3][idx1]=pF1->p[3][idx2];
    pF1Dest->p[7][idx1]=pF1->p[7][idx2];
    pF1Dest->p[6][idx1]=pF1->p[6][idx2];
    pF2Dest->p[3][idx1]=pF2->p[3][idx2];
    pF2Dest->p[7][idx1]=pF2->p[7][idx2];
    pF2Dest->p[6][idx1]=pF2->p[6][idx2];

    idx1=index+(sNy2)*(0);
    idx2=index+(sNy2)*(sNxdp);

    pF1Dest->p[1][idx1]=pF1->p[1][idx2];
    pF1Dest->p[8][idx1]=pF1->p[8][idx2];
    pF1Dest->p[5][idx1]=pF1->p[5][idx2];
    pF2Dest->p[1][idx1]=pF2->p[1][idx2];
    pF2Dest->p[8][idx1]=pF2->p[8][idx2];
    pF2Dest->p[5][idx1]=pF2->p[5][idx2];
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructWW_2_thermal(pop_type *pF1,pop_type *pF1Dest){
  unsigned int idx1, idx2;
  Device_ParallelRepeatUntilIndexIsLessThan(sNy2)
    
    idx1=index+(sNy2)*(sNxdp1);
    idx2=index+(sNy2)*(1);

    pF1Dest->p[3][idx1]=pF1->p[3][idx2];
    pF1Dest->p[7][idx1]=pF1->p[7][idx2];
    pF1Dest->p[6][idx1]=pF1->p[6][idx2];

    idx1=index+(sNy2)*(0);
    idx2=index+(sNy2)*(sNxdp);

    pF1Dest->p[1][idx1]=pF1->p[1][idx2];
    pF1Dest->p[8][idx1]=pF1->p[8][idx2];
    pF1Dest->p[5][idx1]=pF1->p[5][idx2];
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructWW_3(pop_type *pF1Source,pop_type *pF1Dest, pop_type *pF2Source,pop_type *pF2Dest){
  
  unsigned int idx1, idx2;
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i = sNxdp-i; j = sNy-j;
    idx1=j+(sNy2)*i;
    idx2=j+(sNy2)*(i-1);
    pF1Dest->p[1][idx1]=pF1Source->p[1][idx2];
    pF2Dest->p[1][idx1]=pF2Source->p[1][idx2];
    pF1Dest->p[5][idx1]=pF1Source->p[5][idx2-1];
    pF2Dest->p[5][idx1]=pF2Source->p[5][idx2-1];

    idx1=j+(sNy2)*(sNxdp1-i);
    idx2=j-1+(sNy2)*(sNxdp1-i);
    pF1Dest->p[2][idx1]=pF1Source->p[2][idx2];
    pF2Dest->p[2][idx1]=pF2Source->p[2][idx2];
    pF1Dest->p[6][idx1]=pF1Source->p[6][idx2+(sNy2)];
    pF2Dest->p[6][idx1]=pF2Source->p[6][idx2+(sNy2)];

    idx1=(sNy1-j)+(sNy2)*(sNxdp1-i);
    idx2=(sNy1-j)+(sNy2)*((sNxdp1-i)+1);
    pF1Dest->p[3][idx1]=pF1Source->p[3][idx2];
    pF2Dest->p[3][idx1]=pF2Source->p[3][idx2];
    pF1Dest->p[7][idx1]=pF1Source->p[7][idx2+1];
    pF2Dest->p[7][idx1]=pF2Source->p[7][idx2+1];

    idx1=(sNy1-j)+(sNy2)*i;
    idx2=(sNy1-j)+1+(sNy2)*i;
    pF1Dest->p[4][idx1]=pF1Source->p[4][idx2];
    pF2Dest->p[4][idx1]=pF2Source->p[4][idx2];
    pF1Dest->p[8][idx1]=pF1Source->p[8][idx2-(sNy2)];
    pF2Dest->p[8][idx1]=pF2Source->p[8][idx2-(sNy2)];
  Device_ParallelEndRepeat

}

Device_QualKernel void moveplusforcingconstructWW_3_thermal(pop_type *pF1Source,pop_type *pF1Dest){

  unsigned int idx1, idx2;
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i = sNxdp-i; j = sNy-j;
    idx1=j+(sNy2)*i;
    idx2=j+(sNy2)*(i-1);
    pF1Dest->p[1][idx1]=pF1Source->p[1][idx2];
    pF1Dest->p[5][idx1]=pF1Source->p[5][idx2-1];

    idx1=j+(sNy2)*(sNxdp1-i);
    idx2=j-1+(sNy2)*(sNxdp1-i);
    pF1Dest->p[2][idx1]=pF1Source->p[2][idx2];
    pF1Dest->p[6][idx1]=pF1Source->p[6][idx2+(sNy2)];

    idx1=(sNy1-j)+(sNy2)*(sNxdp1-i);
    idx2=(sNy1-j)+(sNy2)*((sNxdp1-i)+1);
    pF1Dest->p[3][idx1]=pF1Source->p[3][idx2];
    pF1Dest->p[7][idx1]=pF1Source->p[7][idx2+1];

    idx1=(sNy1-j)+(sNy2)*i;
    idx2=(sNy1-j)+1+(sNy2)*i;
    pF1Dest->p[4][idx1]=pF1Source->p[4][idx2];
    pF1Dest->p[8][idx1]=pF1Source->p[8][idx2-(sNy2)];
  Device_ParallelEndRepeat

}

Device_QualKernel void moveplusforcingconstructWW_4(pop_type *pF1,pop_type *pF2,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,pop_type *pMemoryup1,pop_type *pMemoryup2,pop_type *pMemorydown1,pop_type *pMemorydown2){

  REAL massapost1,massapost2;
  unsigned int idx1;
  unsigned int i;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)

    i = index+1;
    idx1=sNy+(sNy2)*(i);

    massapost1=pF1->p[0][idx1]+(pMemoryup1->p[2][i]+pMemoryup1->p[5][i]+pMemoryup1->p[6][i])+pF1->p[1][idx1]+pF1->p[3][idx1]+pF1->p[5][idx1]+pF1->p[6][idx1]+pF1->p[2][idx1];

    massapost2=pF2->p[0][idx1]+(pMemoryup2->p[2][i]+pMemoryup2->p[5][i]+pMemoryup2->p[6][i])+pF2->p[1][idx1]+pF2->p[3][idx1]+pF2->p[5][idx1]+pF2->p[6][idx1]+pF2->p[2][idx1];

    pF1->p[4][idx1]=pF1->p[2][idx1]-(v(2.0)/v(3.0))*(v(-0.5)*frcey1[idx1]);

    pF1->p[7][idx1]=pF1->p[5][idx1]+v(0.5)*(pF1->p[1][idx1]-pF1->p[3][idx1])v(-0.5)*(v(-0.5)*frcex1[idx1]+massapost1*sUwallup)-(v(1.0)/v(6.0))*(v(-0.5)*frcey1[idx1]);

    pF1->p[8][idx1]=pF1->p[6][idx1]v(-0.5)*(pF1->p[1][idx1]-pF1->p[3][idx1])+v(0.5)*(v(-0.5)*frcex1[idx1]+massapost1*sUwallup)-(v(1.0)/v(6.0))*(v(-0.5)*frcey1[idx1]);

    pF1->p[0][idx1]=pF1->p[0][idx1]-(pF1->p[4][idx1]+pF1->p[7][idx1]+pF1->p[8][idx1])+(pMemoryup1->p[2][i]+pMemoryup1->p[5][i]+pMemoryup1->p[6][i]);

    pF2->p[4][idx1]=pF2->p[2][idx1]-(v(2.0)/v(3.0))*(v(-0.5)*frcey2[idx1]);

    pF2->p[7][idx1]=pF2->p[5][idx1]+v(0.5)*(pF2->p[1][idx1]-pF2->p[3][idx1])v(-0.5)*(v(-0.5)*frcex2[idx1]+massapost2*sUwallup)-(v(1.0)/v(6.0))*(v(-0.5)*frcey2[idx1]);

    pF2->p[8][idx1]=pF2->p[6][idx1]v(-0.5)*(pF2->p[1][idx1]-pF2->p[3][idx1])+v(0.5)*(v(-0.5)*frcex2[idx1]+massapost2*sUwallup)-(v(1.0)/v(6.0))*(v(-0.5)*frcey2[idx1]);

    pF2->p[0][idx1]=pF2->p[0][idx1]-(pF2->p[4][idx1]+pF2->p[7][idx1]+pF2->p[8][idx1])+(pMemoryup2->p[2][i]+pMemoryup2->p[5][i]+pMemoryup2->p[6][i]);
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructWW_4_thermal(pop_type *pF1,pop_type *pMemoryup1,pop_type *pMemorydown1){

  REAL massapost1;
  unsigned int idx1;
  unsigned int i;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)

    i = index+1;
    idx1=sNy+(sNy2)*(i);

    massapost1=pF1->p[0][idx1]+(pMemoryup1->p[2][i]+pMemoryup1->p[5][i]+pMemoryup1->p[6][i])+pF1->p[1][idx1]+pF1->p[3][idx1]+pF1->p[5][idx1]+pF1->p[6][idx1]+pF1->p[2][idx1];

    pF1->p[4][idx1]=pF1->p[2][idx1];

    pF1->p[7][idx1]=pF1->p[5][idx1]+v(0.5)*(pF1->p[1][idx1]-pF1->p[3][idx1])v(-0.5)*(massapost1*sUwallup);

    pF1->p[8][idx1]=pF1->p[6][idx1]v(-0.5)*(pF1->p[1][idx1]-pF1->p[3][idx1])+v(0.5)*(massapost1*sUwallup);

    pF1->p[0][idx1]=pF1->p[0][idx1]-(pF1->p[4][idx1]+pF1->p[7][idx1]+pF1->p[8][idx1])+(pMemoryup1->p[2][i]+pMemoryup1->p[5][i]+pMemoryup1->p[6][i]);

  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructWW_5(pop_type *pF1,pop_type *pF2,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,pop_type *pMemoryup1,pop_type *pMemoryup2,pop_type *pMemorydown1,pop_type *pMemorydown2){

  REAL massapost1, massapost2;
  unsigned int idx1;
  unsigned int i;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)

    i = index+1;
    idx1=1+(sNy2)*i;

    massapost1=pF1->p[0][idx1]+(pMemorydown1->p[4][i]+pMemorydown1->p[7][i]+pMemorydown1->p[8][i])+pF1->p[1][idx1]+pF1->p[3][idx1]+pF1->p[7][idx1]+pF1->p[8][idx1]+pF1->p[4][idx1];

    massapost2=pF2->p[0][idx1]+(pMemorydown2->p[4][i]+pMemorydown2->p[7][i]+pMemorydown2->p[8][i])+pF2->p[1][idx1]+pF2->p[3][idx1]+pF2->p[7][idx1]+pF2->p[8][idx1]+pF2->p[4][idx1];

    pF1->p[2][idx1]=pF1->p[4][idx1]+(v(2.0)/v(3.0))*(v(-0.5)*frcey1[idx1]);

    pF1->p[5][idx1]=pF1->p[7][idx1]v(-0.5)*(pF1->p[1][idx1]-pF1->p[3][idx1])+v(0.5)*(v(-0.5)*frcex1[idx1]+massapost1*sUwalldown)+(v(1.0)/v(6.0))*(v(-0.5)*frcey1[idx1]);

    pF1->p[6][idx1]=pF1->p[8][idx1]+v(0.5)*(pF1->p[1][idx1]-pF1->p[3][idx1])v(-0.5)*(v(-0.5)*frcex1[idx1]+massapost1*sUwalldown)+(v(1.0)/v(6.0))*(v(-0.5)*frcey1[idx1]);

    pF1->p[0][idx1]=pF1->p[0][idx1]-(pF1->p[2][idx1]+pF1->p[5][idx1]+pF1->p[6][idx1])+(pMemorydown1->p[4][i]+pMemorydown1->p[7][i]+pMemorydown1->p[8][i]);

    pF2->p[2][idx1]=pF2->p[4][idx1]+(v(2.0)/v(3.0))*(v(-0.5)*frcey2[idx1]);

    pF2->p[5][idx1]=pF2->p[7][idx1]v(-0.5)*(pF2->p[1][idx1]-pF2->p[3][idx1])+v(0.5)*(v(-0.5)*frcex2[idx1]+massapost2*sUwalldown)+(v(1.0)/v(6.0))*(v(-0.5)*frcey2[idx1]);

    pF2->p[6][idx1]=pF2->p[8][idx1]+v(0.5)*(pF2->p[1][idx1]-pF2->p[3][idx1])v(-0.5)*(v(-0.5)*frcex2[idx1]+massapost2*sUwalldown)+(v(1.0)/v(6.0))*(v(-0.5)*frcey2[idx1]);

    pF2->p[0][idx1]=pF2->p[0][idx1]-(pF2->p[2][idx1]+pF2->p[5][idx1]+pF2->p[6][idx1])+(pMemorydown2->p[4][i]+pMemorydown2->p[7][i]+pMemorydown2->p[8][i]);
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructWW_5_thermal(pop_type *pF1,pop_type *pMemoryup1,pop_type *pMemorydown1){

  REAL massapost1;
  unsigned int idx1;
  unsigned int i;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp)

    i = index+1;
    idx1=1+(sNy2)*i;

    massapost1=pF1->p[0][idx1]+(pMemorydown1->p[4][i]+pMemorydown1->p[7][i]+pMemorydown1->p[8][i])+pF1->p[1][idx1]+pF1->p[3][idx1]+pF1->p[7][idx1]+pF1->p[8][idx1]+pF1->p[4][idx1];

    pF1->p[2][idx1]=pF1->p[4][idx1];

    pF1->p[5][idx1]=pF1->p[7][idx1]v(-0.5)*(pF1->p[1][idx1]-pF1->p[3][idx1])+v(0.5)*(massapost1*sUwalldown);

    pF1->p[6][idx1]=pF1->p[8][idx1]+v(0.5)*(pF1->p[1][idx1]-pF1->p[3][idx1])v(-0.5)*(massapost1*sUwalldown);

    pF1->p[0][idx1]=pF1->p[0][idx1]-(pF1->p[2][idx1]+pF1->p[5][idx1]+pF1->p[6][idx1])+(pMemorydown1->p[4][i]+pMemorydown1->p[7][i]+pMemorydown1->p[8][i]);

  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructobstacle_2(pop_type *pF1Source,pop_type *pF2Source, pop_type *pF1Dest,pop_type *pF2Dest){

  unsigned int idx1, idx2, idx3, idx4;
  unsigned int i;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)

    idx1=(sNy2)*index;
    idx3=1+(sNy2)*index;
    pF1Dest->p[2][idx1]=pF1Source->p[4][idx3];
    pF2Dest->p[2][idx1]=pF2Source->p[4][idx3];
  Device_ParallelEndRepeat

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp1)
    i=index;
    idx1=(sNy2)*index;
    idx2=1+(sNy2)*(i+1);
    pF1Dest->p[5][idx1]=pF1Source->p[7][idx2];
    pF2Dest->p[5][idx1]=pF2Source->p[7][idx2];

  Device_ParallelEndRepeat

  Device_ParallelRepeatStartingFromAndUntilIndexIsLessThan(1,sNxdp2)
    i=index;
    idx1=(sNy2)*index;
    idx4=1+(sNy2)*(i-1);
    pF1Dest->p[6][idx1]=pF1Source->p[8][idx4];
    pF2Dest->p[6][idx1]=pF2Source->p[8][idx4];
  Device_ParallelEndRepeat

}

Device_QualKernel void moveplusforcingconstructobstacle_2_thermal(pop_type *pF1Source, pop_type *pF1Dest){

  unsigned int idx1, idx2, idx3, idx4;
  unsigned int i;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)

    idx1=(sNy2)*index;
    idx3=1+(sNy2)*index;
    pF1Dest->p[2][idx1]=pF1Source->p[4][idx3];
  Device_ParallelEndRepeat

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp1)
    i=index;
    idx1=(sNy2)*index;
    idx2=1+(sNy2)*(i+1);
    pF1Dest->p[5][idx1]=pF1Source->p[7][idx2];

  Device_ParallelEndRepeat

  Device_ParallelRepeatStartingFromAndUntilIndexIsLessThan(1,sNxdp2)
    i=index;
    idx1=(sNy2)*index;
    idx4=1+(sNy2)*(i-1);
    pF1Dest->p[6][idx1]=pF1Source->p[8][idx4];
  Device_ParallelEndRepeat

}

Device_QualKernel void moveplusforcingconstructobstacle_3UpperWall(pop_type *pF1Source,pop_type *pF2Source, pop_type *pF1Dest,pop_type *pF2Dest){

  unsigned int idx1, idx2, idx3, idx4;
  unsigned int i;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)

    idx1=sNy1 + (sNy2)*index;
    idx3=sNy + (sNy2)*index;
    pF1Dest->p[4][idx1]=pF1Source->p[2][idx3];
    pF2Dest->p[4][idx1]=pF2Source->p[2][idx3];
  Device_ParallelEndRepeat
  Device_ParallelRepeatStartingFromAndUntilIndexIsLessThan(1,sNxdp2)
    i=index;
    idx1=sNy1 + (sNy2)*index;
    idx2=sNy + (sNy2)*(i-1);
    pF1Dest->p[7][idx1]=pF1Source->p[5][idx2];
    pF2Dest->p[7][idx1]=pF2Source->p[5][idx2];

  Device_ParallelEndRepeat

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp1)
    i=index;
    idx1=sNy1 + (sNy2)*index;
    idx4=sNy + (sNy2)*(i+1);
    pF1Dest->p[8][idx1]=pF1Source->p[6][idx4];
    pF2Dest->p[8][idx1]=pF2Source->p[6][idx4];
  Device_ParallelEndRepeat

}

Device_QualKernel void moveplusforcingconstructobstacle_3UpperWall_thermal(pop_type *pF1Source, pop_type *pF1Dest){

  unsigned int idx1, idx2, idx3, idx4;
  unsigned int i;

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)

    idx1=sNy1 + (sNy2)*index;
    idx3=sNy + (sNy2)*index;
    pF1Dest->p[4][idx1]=pF1Source->p[2][idx3];
  Device_ParallelEndRepeat

  Device_ParallelRepeatStartingFromAndUntilIndexIsLessThan(1,sNxdp2)
    i=index;
    idx1=sNy1 + (sNy2)*index;
    idx2=sNy + (sNy2)*(i-1);
    pF1Dest->p[7][idx1]=pF1Source->p[5][idx2];
  Device_ParallelEndRepeat

  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp1)
    i=index;
    idx1=sNy1 + (sNy2)*index;
    idx4=sNy + (sNy2)*(i+1);
    pF1Dest->p[8][idx1]=pF1Source->p[6][idx4];
  Device_ParallelEndRepeat

}

Device_QualKernel void obstacle(pop_type *pF1Source,pop_type *pF2Source,
                                pop_type *pF1Dest,pop_type *pF2Dest, int *flag){

  unsigned int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
  unsigned int plusx1, plusy1, minusx1, minusy1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;

    plusx1=i+1;
    minusx1=i-1;
    plusy1=j+1;
    minusy1=j-1;

    idx0=j+(sNy2)*i;

    idx1=j+(sNy2)*plusx1;
    idx2=plusy1+(sNy2)*i;
    idx3=j+(sNy2)*minusx1;
    idx4=minusy1+(sNy2)*i;

    idx5=plusy1+(sNy2)*plusx1;
    idx6=plusy1+(sNy2)*minusx1;
    idx7=minusy1+(sNy2)*minusx1;
    idx8=minusy1+(sNy2)*plusx1;

    if(!flag[idx0]){
      pF1Dest->p[3][idx0]=flag[idx3]*pF1Source->p[1][idx3]+(1-flag[idx3])*pF1Dest->p[3][idx0];
      pF1Dest->p[4][idx0]=flag[idx4]*pF1Source->p[2][idx4]+(1-flag[idx4])*pF1Dest->p[4][idx0];
      pF1Dest->p[1][idx0]=flag[idx1]*pF1Source->p[3][idx1]+(1-flag[idx1])*pF1Dest->p[1][idx0];
      pF1Dest->p[2][idx0]=flag[idx2]*pF1Source->p[4][idx2]+(1-flag[idx2])*pF1Dest->p[2][idx0];
      pF1Dest->p[7][idx0]=flag[idx7]*pF1Source->p[5][idx7]+(1-flag[idx7])*pF1Dest->p[7][idx0];
      pF1Dest->p[8][idx0]=flag[idx8]*pF1Source->p[6][idx8]+(1-flag[idx8])*pF1Dest->p[8][idx0];
      pF1Dest->p[5][idx0]=flag[idx5]*pF1Source->p[7][idx5]+(1-flag[idx5])*pF1Dest->p[5][idx0];
      pF1Dest->p[6][idx0]=flag[idx6]*pF1Source->p[8][idx6]+(1-flag[idx6])*pF1Dest->p[6][idx0];

      pF2Dest->p[3][idx0]=flag[idx3]*pF2Source->p[1][idx3]+(1-flag[idx3])*pF2Dest->p[3][idx0];
      pF2Dest->p[4][idx0]=flag[idx4]*pF2Source->p[2][idx4]+(1-flag[idx4])*pF2Dest->p[4][idx0];
      pF2Dest->p[1][idx0]=flag[idx1]*pF2Source->p[3][idx1]+(1-flag[idx1])*pF2Dest->p[1][idx0];
      pF2Dest->p[2][idx0]=flag[idx2]*pF2Source->p[4][idx2]+(1-flag[idx2])*pF2Dest->p[2][idx0];
      pF2Dest->p[7][idx0]=flag[idx7]*pF2Source->p[5][idx7]+(1-flag[idx7])*pF2Dest->p[7][idx0];
      pF2Dest->p[8][idx0]=flag[idx8]*pF2Source->p[6][idx8]+(1-flag[idx8])*pF2Dest->p[8][idx0];
      pF2Dest->p[5][idx0]=flag[idx5]*pF2Source->p[7][idx5]+(1-flag[idx5])*pF2Dest->p[5][idx0];
      pF2Dest->p[6][idx0]=flag[idx6]*pF2Source->p[8][idx6]+(1-flag[idx6])*pF2Dest->p[6][idx0];

    }
  Device_ParallelEndRepeat

  return;
}

Device_QualKernel void obstacle_thermal(pop_type *pF1Source,pop_type *pF1Dest, int *flag){

  unsigned int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
  unsigned int plusx1, plusy1, minusx1, minusy1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;

    plusx1=i+1;
    minusx1=i-1;
    plusy1=j+1;
    minusy1=j-1;

    idx0=j+(sNy2)*i;

    idx1=j+(sNy2)*plusx1;
    idx2=plusy1+(sNy2)*i;
    idx3=j+(sNy2)*minusx1;
    idx4=minusy1+(sNy2)*i;

    idx5=plusy1+(sNy2)*plusx1;
    idx6=plusy1+(sNy2)*minusx1;
    idx7=minusy1+(sNy2)*minusx1;
    idx8=minusy1+(sNy2)*plusx1;

    if(!flag[idx0]){
      pF1Dest->p[3][idx0]=flag[idx3]*pF1Source->p[1][idx3]+(1-flag[idx3])*pF1Dest->p[3][idx0];
      pF1Dest->p[4][idx0]=flag[idx4]*pF1Source->p[2][idx4]+(1-flag[idx4])*pF1Dest->p[4][idx0];
      pF1Dest->p[1][idx0]=flag[idx1]*pF1Source->p[3][idx1]+(1-flag[idx1])*pF1Dest->p[1][idx0];
      pF1Dest->p[2][idx0]=flag[idx2]*pF1Source->p[4][idx2]+(1-flag[idx2])*pF1Dest->p[2][idx0];
      pF1Dest->p[7][idx0]=flag[idx7]*pF1Source->p[5][idx7]+(1-flag[idx7])*pF1Dest->p[7][idx0];
      pF1Dest->p[8][idx0]=flag[idx8]*pF1Source->p[6][idx8]+(1-flag[idx8])*pF1Dest->p[8][idx0];
      pF1Dest->p[5][idx0]=flag[idx5]*pF1Source->p[7][idx5]+(1-flag[idx5])*pF1Dest->p[5][idx0];
      pF1Dest->p[6][idx0]=flag[idx6]*pF1Source->p[8][idx6]+(1-flag[idx6])*pF1Dest->p[6][idx0];
    }
  Device_ParallelEndRepeat

  return;
}

Device_QualKernel void obstacleUpperWall(pop_type *pF1Source,pop_type *pF2Source,
                                pop_type *pF1Dest,pop_type *pF2Dest, int *flag){

  unsigned int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
  unsigned int plusx1, plusy1, minusx1, minusy1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;

    plusx1=i+1;
    minusx1=i-1;
    plusy1=j+1;
    minusy1=j-1;

    idx0=j+(sNy2)*i;

    idx1=j+(sNy2)*plusx1;
    idx2=plusy1+(sNy2)*i;
    idx3=j+(sNy2)*minusx1;
    idx4=minusy1+(sNy2)*i;

    idx5=plusy1+(sNy2)*plusx1;
    idx6=plusy1+(sNy2)*minusx1;
    idx7=minusy1+(sNy2)*minusx1;
    idx8=minusy1+(sNy2)*plusx1;

    if(!flag[idx0]){
      pF1Dest->p[3][idx0]=flag[idx3]*pF1Source->p[1][idx3]+(1-flag[idx3])*pF1Dest->p[3][idx0];
      pF1Dest->p[4][idx0]=flag[idx2]*pF1Source->p[2][idx4]+(1-flag[idx2])*pF1Dest->p[4][idx0];
      pF1Dest->p[1][idx0]=flag[idx1]*pF1Source->p[3][idx1]+(1-flag[idx1])*pF1Dest->p[1][idx0];
      pF1Dest->p[2][idx0]=flag[idx2]*pF1Source->p[4][idx2]+(1-flag[idx2])*pF1Dest->p[2][idx0];
      pF1Dest->p[7][idx0]=flag[idx7]*pF1Source->p[5][idx7]+(1-flag[idx7])*pF1Dest->p[7][idx0];
      pF1Dest->p[8][idx0]=flag[idx8]*pF1Source->p[6][idx8]+(1-flag[idx8])*pF1Dest->p[8][idx0];
      pF1Dest->p[5][idx0]=flag[idx5]*pF1Source->p[7][idx5]+(1-flag[idx5])*pF1Dest->p[5][idx0];
      pF1Dest->p[6][idx0]=flag[idx6]*pF1Source->p[8][idx6]+(1-flag[idx6])*pF1Dest->p[6][idx0];

      pF2Dest->p[3][idx0]=flag[idx3]*pF2Source->p[1][idx3]+(1-flag[idx3])*pF2Dest->p[3][idx0];
      pF2Dest->p[4][idx0]=flag[idx2]*pF2Source->p[2][idx4]+(1-flag[idx2])*pF2Dest->p[4][idx0];
      pF2Dest->p[1][idx0]=flag[idx1]*pF2Source->p[3][idx1]+(1-flag[idx1])*pF2Dest->p[1][idx0];
      pF2Dest->p[2][idx0]=flag[idx2]*pF2Source->p[4][idx2]+(1-flag[idx2])*pF2Dest->p[2][idx0];
      pF2Dest->p[7][idx0]=flag[idx7]*pF2Source->p[5][idx7]+(1-flag[idx7])*pF2Dest->p[7][idx0];
      pF2Dest->p[8][idx0]=flag[idx8]*pF2Source->p[6][idx8]+(1-flag[idx8])*pF2Dest->p[8][idx0];
      pF2Dest->p[5][idx0]=flag[idx5]*pF2Source->p[7][idx5]+(1-flag[idx5])*pF2Dest->p[5][idx0];
      pF2Dest->p[6][idx0]=flag[idx6]*pF2Source->p[8][idx6]+(1-flag[idx6])*pF2Dest->p[6][idx0];

    }
  Device_ParallelEndRepeat

  return;
}

Device_QualKernel void obstacleUpperWall_thermal(pop_type *pF1Source,pop_type *pF1Dest, int *flag){

  unsigned int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
  unsigned int plusx1, plusy1, minusx1, minusy1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;

    plusx1=i+1;
    minusx1=i-1;
    plusy1=j+1;
    minusy1=j-1;

    idx0=j+(sNy2)*i;

    idx1=j+(sNy2)*plusx1;
    idx2=plusy1+(sNy2)*i;
    idx3=j+(sNy2)*minusx1;
    idx4=minusy1+(sNy2)*i;

    idx5=plusy1+(sNy2)*plusx1;
    idx6=plusy1+(sNy2)*minusx1;
    idx7=minusy1+(sNy2)*minusx1;
    idx8=minusy1+(sNy2)*plusx1;

    if(!flag[idx0]){
      pF1Dest->p[3][idx0]=flag[idx3]*pF1Source->p[1][idx3]+(1-flag[idx3])*pF1Dest->p[3][idx0];
      pF1Dest->p[4][idx0]=flag[idx2]*pF1Source->p[2][idx4]+(1-flag[idx2])*pF1Dest->p[4][idx0];
      pF1Dest->p[1][idx0]=flag[idx1]*pF1Source->p[3][idx1]+(1-flag[idx1])*pF1Dest->p[1][idx0];
      pF1Dest->p[2][idx0]=flag[idx2]*pF1Source->p[4][idx2]+(1-flag[idx2])*pF1Dest->p[2][idx0];
      pF1Dest->p[7][idx0]=flag[idx7]*pF1Source->p[5][idx7]+(1-flag[idx7])*pF1Dest->p[7][idx0];
      pF1Dest->p[8][idx0]=flag[idx8]*pF1Source->p[6][idx8]+(1-flag[idx8])*pF1Dest->p[8][idx0];
      pF1Dest->p[5][idx0]=flag[idx5]*pF1Source->p[7][idx5]+(1-flag[idx5])*pF1Dest->p[5][idx0];
      pF1Dest->p[6][idx0]=flag[idx6]*pF1Source->p[8][idx6]+(1-flag[idx6])*pF1Dest->p[6][idx0];
    }
  Device_ParallelEndRepeat

  return;
}

void moveplusforcingconstructWW(pop_type *pF1Source,pop_type *pF1Dest,
                                pop_type *pF2Source,pop_type *pF2Dest,
                                pop_type *pHostF1Source,pop_type *pHostF1Dest,
                                pop_type *pHostF2Source,pop_type *pHostF2Dest,
                                REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,
                                REAL *frcex2,REAL *frcey2,REAL *psi1self,REAL *psi2self,
                                pop_type *pMemoryup1,pop_type *pMemoryup2,
                                pop_type *pMemorydown1,pop_type *pMemorydown2,
                                REAL uwalluptd, REAL uwalldowntd,REAL *utot,REAL *vtot,
				REAL *tau){

  DeviceEnableLoadREALValuesForBlock();

  Device_ExecuteKernel(moveplusforcingconstructWW_1)(pF1Source, pF2Source, pMemoryup1, pMemoryup2, pMemorydown1, pMemorydown2);
   Device_Synchronize();

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Dest->p[i], pHostF1Source->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Dest->p[i], pHostF2Source->p[i], REAL, (nxdp+2)*(ny+2));
  }

   if(monogpu) {
  Device_ExecuteKernel(moveplusforcingconstructWW_2)(pF1Source, pF2Source, pF1Dest, pF2Dest);
   Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Source->p[i], pHostF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_ExecuteKernel(moveplusforcingconstructWW_3)(pF1Source, pF1Dest, pF2Source, pF2Dest);
   Device_Synchronize();

  rhocomp(rho1,pF1Dest);
  rhocomp(rho2,pF2Dest);

  setrhobc1(rho1,pMemoryup1,pMemorydown1,pF1Dest);
  setrhobc2(rho2,pMemoryup2,pMemorydown2,pF2Dest);
  
  forcingconstructWW(rho1,rho2,temperature,frcex1,frcey1,frcex2,frcey2,psi1self,psi2self,utot,vtot,tau);

  Device_Synchronize();

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Source->p[i], pHostF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_SafeLoadREALValueOnDevice(sUwallup, uwalluptd);
  Device_SafeLoadREALValueOnDevice(sUwalldown, uwalldowntd);

  Device_ExecuteKernel(moveplusforcingconstructWW_4)(pF1Source, pF2Source, frcex1, frcey1, frcex2, frcey2, pMemoryup1, pMemoryup2, pMemorydown1, pMemorydown2);
   Device_Synchronize();


  Device_ExecuteKernel(moveplusforcingconstructWW_5)(pF1Source, pF2Source, frcex1, frcey1, frcex2, frcey2, pMemoryup1, pMemoryup2, pMemorydown1, pMemorydown2);
   Device_Synchronize();

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Dest->p[i], pHostF1Source->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Dest->p[i], pHostF2Source->p[i], REAL, (nxdp+2)*(ny+2));
  }
}

void moveconstructWW_thermal(pop_type *pF1Source,pop_type *pF1Dest,
			     pop_type *pHostF1Source,pop_type *pHostF1Dest,
			     REAL *rho1,pop_type *pBndUpG,pop_type *pBndDownG,
			     REAL Tup, REAL Tdown, int initcond){

  Device_ExecuteKernel(moveplusforcingconstructWW_1_thermal)(pF1Source, pBndUpG, pBndDownG, Tup, Tdown, initcond);
  Device_Synchronize();

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Dest->p[i], pHostF1Source->p[i], REAL, (nxdp+2)*(ny+2));
  }

   if(monogpu) {
     Device_ExecuteKernel(moveplusforcingconstructWW_2_thermal)(pF1Source, pF1Dest);
     Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_ExecuteKernel(moveplusforcingconstructWW_3_thermal)(pF1Source, pF1Dest);
  Device_Synchronize();

}


void moveplusforcingconstructobstacle(pop_type *pF1Source,pop_type *pF1Dest,
				      pop_type *pF2Source,pop_type *pF2Dest,
				      pop_type *pHostF1Source,pop_type *pHostF1Dest,
				      pop_type *pHostF2Source,pop_type *pHostF2Dest,
				      REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,
				      REAL *frcex2,REAL *frcey2,REAL *psi1self,REAL *psi2self,
				      pop_type *pMemoryup1,pop_type *pMemoryup2,
				      pop_type *pMemorydown1,pop_type *pMemorydown2,
				      REAL uwalluptd, REAL uwalldowntd,
				      REAL *utot,REAL *vtot, int *flag,
				      REAL *tau){
  
  DeviceEnableLoadREALValuesForBlock();

  Device_ExecuteKernel(moveplusforcingconstructobstacle_1)(pF1Source, pF2Source, pMemoryup1, pMemoryup2, pMemorydown1, pMemorydown2);
  Device_Synchronize();

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Dest->p[i], pHostF1Source->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Dest->p[i], pHostF2Source->p[i], REAL, (nxdp+2)*(ny+2));
  }

  if(monogpu) {
    Device_ExecuteKernel(moveplusforcingconstructWW_2)(pF1Source, pF2Source, pF1Dest, pF2Dest);
    Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Source->p[i], pHostF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_ExecuteKernel(moveplusforcingconstructobstacle_2)(pF1Source, pF2Source,
                                                           pF1Dest, pF2Dest);
  Device_Synchronize();

  if(roughWallUp){
    Device_ExecuteKernel(moveplusforcingconstructobstacle_3UpperWall)(pF1Source, pF2Source,
                                                                      pF1Dest, pF2Dest);
    Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Source->p[i], pHostF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_ExecuteKernel(obstacle)(pF1Source, pF2Source, pF1Dest, pF2Dest, flag);
  Device_Synchronize();

  if(roughWallUp){
    Device_ExecuteKernel(obstacleUpperWall)(pF1Source, pF2Source, pF1Dest, pF2Dest, flag);
    Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Source->p[i], pHostF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_ExecuteKernel(moveplusforcingconstructWW_3)(pF1Source, pF1Dest, pF2Source, pF2Dest);
  Device_Synchronize();

  rhocomp(rho1,pF1Dest);
  rhocomp(rho2,pF2Dest);

  setrhobc1obstacle(rho1,pMemoryup1,pMemorydown1,pF1Dest,flag);
  setrhobc2obstacle(rho2,pMemoryup2,pMemorydown2,pF2Dest,flag);

  forcingconstructWW(rho1,rho2,temperature,frcex1,frcey1,frcex2,frcey2,psi1self,psi2self,utot,vtot,tau);

  Device_Synchronize();

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Source->p[i], pHostF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_SafeLoadREALValueOnDevice(sUwallup, uwalluptd);
  Device_SafeLoadREALValueOnDevice(sUwalldown, uwalldowntd);

  if(roughWallUp==0){
    Device_ExecuteKernel(moveplusforcingconstructWW_4)(pF1Source, pF2Source, frcex1, frcey1, frcex2, frcey2, pMemoryup1, pMemoryup2, pMemorydown1, pMemorydown2);
    Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Dest->p[i], pHostF1Source->p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryCopyDevice(pHostF2Dest->p[i], pHostF2Source->p[i], REAL, (nxdp+2)*(ny+2));
  }
}

void moveconstructobstacle_thermal(pop_type *pF1Source,pop_type *pF1Dest,
				   pop_type *pHostF1Source,pop_type *pHostF1Dest,
				   REAL *rho1, pop_type *pBndUpG, pop_type *pBndDownG,
				   REAL Tup, REAL Tdown, int initcond, int *flag){
  

  Device_ExecuteKernel(moveplusforcingconstructWW_1_thermal)(pF1Source, pBndUpG, pBndDownG, Tup, Tdown, initcond);
  Device_Synchronize();

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Dest->p[i], pHostF1Source->p[i], REAL, (nxdp+2)*(ny+2));
  }

  if(monogpu) {
    Device_ExecuteKernel(moveplusforcingconstructWW_2_thermal)(pF1Source, pF1Dest);
    Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_ExecuteKernel(moveplusforcingconstructobstacle_2_thermal)(pF1Source,pF1Dest);
  Device_Synchronize();

  if(roughWallUp){
    Device_ExecuteKernel(moveplusforcingconstructobstacle_3UpperWall_thermal)(pF1Source, pF1Dest);
    Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_ExecuteKernel(obstacle_thermal)(pF1Source, pF1Dest, flag);
  Device_Synchronize();

  if(roughWallUp){
    Device_ExecuteKernel(obstacleUpperWall_thermal)(pF1Source, pF1Dest, flag);
    Device_Synchronize();
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
  }

  Device_ExecuteKernel(moveplusforcingconstructWW_3_thermal)(pF1Source, pF1Dest);
  Device_Synchronize();

}

Device_QualKernel void moveplusforcingconstructPBC_1(pop_type *pF1,pop_type *pF2, pop_type *pF1Dest,pop_type *pF2Dest){
  unsigned int idx1, idx2;
  Device_ParallelRepeatUntilIndexIsLessThan(sNy2)

    idx1=index+(sNy2)*(sNxdp1);
    idx2=index+(sNy2)*(1);

    pF1Dest->p[3][idx1]=pF1->p[3][idx2];
    pF1Dest->p[7][idx1]=pF1->p[7][idx2];
    pF1Dest->p[6][idx1]=pF1->p[6][idx2];
    pF2Dest->p[3][idx1]=pF2->p[3][idx2];
    pF2Dest->p[7][idx1]=pF2->p[7][idx2];
    pF2Dest->p[6][idx1]=pF2->p[6][idx2];

    idx1=index+(sNy2)*(0);
    idx2=index+(sNy2)*(sNxdp);

    pF1Dest->p[1][idx1]=pF1->p[1][idx2];
    pF1Dest->p[8][idx1]=pF1->p[8][idx2];
    pF1Dest->p[5][idx1]=pF1->p[5][idx2];
    pF2Dest->p[1][idx1]=pF2->p[1][idx2];
    pF2Dest->p[8][idx1]=pF2->p[8][idx2];
    pF2Dest->p[5][idx1]=pF2->p[5][idx2];
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructPBC_1_thermal(pop_type *pF1, pop_type *pF1Dest){
  unsigned int idx1, idx2;
  Device_ParallelRepeatUntilIndexIsLessThan(sNy2)
    
    idx1=index+(sNy2)*(sNxdp1);
    idx2=index+(sNy2)*(1);

    pF1Dest->p[3][idx1]=pF1->p[3][idx2];
    pF1Dest->p[7][idx1]=pF1->p[7][idx2];
    pF1Dest->p[6][idx1]=pF1->p[6][idx2];

    idx1=index+(sNy2)*(0);
    idx2=index+(sNy2)*(sNxdp);

    pF1Dest->p[1][idx1]=pF1->p[1][idx2];
    pF1Dest->p[8][idx1]=pF1->p[8][idx2];
    pF1Dest->p[5][idx1]=pF1->p[5][idx2];
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructPBC_2(pop_type *pF1,pop_type *pF2, pop_type *pF1Dest,pop_type *pF2Dest){
  unsigned int idx1, idx2;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)
    
    idx1=0+(sNy2)*index;
    idx2=sNy+(sNy2)*index;

    pF1Dest->p[2][idx1]=pF1->p[2][idx2];
    pF1Dest->p[5][idx1]=pF1->p[5][idx2];
    pF1Dest->p[6][idx1]=pF1->p[6][idx2];
    pF2Dest->p[2][idx1]=pF2->p[2][idx2];
    pF2Dest->p[5][idx1]=pF2->p[5][idx2];
    pF2Dest->p[6][idx1]=pF2->p[6][idx2];

    idx1=sNy+1+(sNy2)*index;
    idx2=1+(sNy2)*index;

    pF1Dest->p[4][idx1]=pF1->p[4][idx2];
    pF1Dest->p[7][idx1]=pF1->p[7][idx2];
    pF1Dest->p[8][idx1]=pF1->p[8][idx2];
    pF2Dest->p[4][idx1]=pF2->p[4][idx2];
    pF2Dest->p[7][idx1]=pF2->p[7][idx2];
    pF2Dest->p[8][idx1]=pF2->p[8][idx2];
  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructPBC_2_thermal(pop_type *pF1, pop_type *pF1Dest){
  unsigned int idx1, idx2;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2)

    idx1=0+(sNy2)*index;
    idx2=sNy+(sNy2)*index;

    pF1Dest->p[2][idx1]=pF1->p[2][idx2];
    pF1Dest->p[5][idx1]=pF1->p[5][idx2];
    pF1Dest->p[6][idx1]=pF1->p[6][idx2];

    idx1=sNy+1+(sNy2)*index;
    idx2=1+(sNy2)*index;

    pF1Dest->p[4][idx1]=pF1->p[4][idx2];
    pF1Dest->p[7][idx1]=pF1->p[7][idx2];
    pF1Dest->p[8][idx1]=pF1->p[8][idx2];

  Device_ParallelEndRepeat
}

Device_QualKernel void moveplusforcingconstructPBC_3(pop_type *pF1,pop_type *pF2, pop_type *pF1Dest,pop_type *pF2Dest){

  unsigned int idx1, idx2;
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i = sNxdp-i; j = sNy-j;
    idx1=j+(sNy2)*i;
    idx2=j+(sNy2)*(i-1);
    pF1Dest->p[1][idx1]=pF1->p[1][idx2];
    pF2Dest->p[1][idx1]=pF2->p[1][idx2];
    pF1Dest->p[5][idx1]=pF1->p[5][idx2-1];
    pF2Dest->p[5][idx1]=pF2->p[5][idx2-1];

    idx1=j+(sNy2)*(sNxdp1-i);
    idx2=j-1+(sNy2)*(sNxdp1-i);
    pF1Dest->p[2][idx1]=pF1->p[2][idx2];
    pF2Dest->p[2][idx1]=pF2->p[2][idx2];
    pF1Dest->p[6][idx1]=pF1->p[6][idx2+(sNy2)];
    pF2Dest->p[6][idx1]=pF2->p[6][idx2+(sNy2)];

    idx1=(sNy1-j)+(sNy2)*(sNxdp1-i);
    idx2=(sNy1-j)+(sNy2)*((sNxdp1-i)+1);
    pF1Dest->p[3][idx1]=pF1->p[3][idx2];
    pF2Dest->p[3][idx1]=pF2->p[3][idx2];
    pF1Dest->p[7][idx1]=pF1->p[7][idx2+1];
    pF2Dest->p[7][idx1]=pF2->p[7][idx2+1];

    idx1=(sNy1-j)+(sNy2)*i;
    idx2=(sNy1-j)+1+(sNy2)*i;
    pF1Dest->p[4][idx1]=pF1->p[4][idx2];
    pF2Dest->p[4][idx1]=pF2->p[4][idx2];
    pF1Dest->p[8][idx1]=pF1->p[8][idx2-(sNy2)];
    pF2Dest->p[8][idx1]=pF2->p[8][idx2-(sNy2)];
  Device_ParallelEndRepeat

}

Device_QualKernel void moveplusforcingconstructPBC_3_thermal(pop_type *pF1, pop_type *pF1Dest){

  unsigned int idx1, idx2;
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i = sNxdp-i; j = sNy-j;
    idx1=j+(sNy2)*i;
    idx2=j+(sNy2)*(i-1);
    pF1Dest->p[1][idx1]=pF1->p[1][idx2];
    pF1Dest->p[5][idx1]=pF1->p[5][idx2-1];

    idx1=j+(sNy2)*(sNxdp1-i);
    idx2=j-1+(sNy2)*(sNxdp1-i);
    pF1Dest->p[2][idx1]=pF1->p[2][idx2];
    pF1Dest->p[6][idx1]=pF1->p[6][idx2+(sNy2)];

    idx1=(sNy1-j)+(sNy2)*(sNxdp1-i);
    idx2=(sNy1-j)+(sNy2)*((sNxdp1-i)+1);
    pF1Dest->p[3][idx1]=pF1->p[3][idx2];
    pF1Dest->p[7][idx1]=pF1->p[7][idx2+1];

    idx1=(sNy1-j)+(sNy2)*i;
    idx2=(sNy1-j)+1+(sNy2)*i;
    pF1Dest->p[4][idx1]=pF1->p[4][idx2];
    pF1Dest->p[8][idx1]=pF1->p[8][idx2-(sNy2)];
  Device_ParallelEndRepeat

}

void moveplusforcingconstructPBC(pop_type *pF1Source,pop_type *pF1Dest,
                                 pop_type *pF2Source,pop_type *pF2Dest,
                                 pop_type *pHostF1Source,pop_type *pHostF1Dest,
                                 pop_type *pHostF2Source,pop_type *pHostF2Dest,
                                 REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,
                                 REAL *frcex2,REAL *frcey2,
                                 REAL *psi1self,REAL *psi2self,REAL *tau){

  DeviceEnableLoadREALValuesForBlock();

   for(i=0; i<npop; i++) {
     Device_SafeMemoryCopyDevice(pHostF1Dest->p[i], pHostF1Source->p[i], REAL, (nxdp+2)*(ny+2));
     Device_SafeMemoryCopyDevice(pHostF2Dest->p[i], pHostF2Source->p[i], REAL, (nxdp+2)*(ny+2));
   }
   if(monogpu) {
   Device_ExecuteKernel(moveplusforcingconstructPBC_1)(pF1Source, pF2Source, pF1Dest, pF2Dest);
    Device_Synchronize();
   }

   for(i=0; i<npop; i++) {
     Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
     Device_SafeMemoryCopyDevice(pHostF2Source->p[i], pHostF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
   }

   Device_ExecuteKernel(moveplusforcingconstructPBC_2)(pF1Source, pF2Source, pF1Dest, pF2Dest);
    Device_Synchronize();

   for(i=0; i<npop; i++) {
     Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
     Device_SafeMemoryCopyDevice(pHostF2Source->p[i], pHostF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
   }
    Device_Synchronize();

   Device_ExecuteKernel(moveplusforcingconstructPBC_3)(pF1Source, pF2Source, pF1Dest, pF2Dest);
    Device_Synchronize();

   rhocomp(rho1,pF1Dest);
   rhocomp(rho2,pF2Dest);

   forcingconstructPBC(rho1,rho2,temperature,frcex1,frcey1,frcex2,frcey2,psi1self,psi2self, tau);

}

void moveconstructPBC_thermal(pop_type *pF1Source,pop_type *pF1Dest,
			      pop_type *pHostF1Source,pop_type *pHostF1Dest,
			      REAL *rho1){

   for(i=0; i<npop; i++) {
     Device_SafeMemoryCopyDevice(pHostF1Dest->p[i], pHostF1Source->p[i], REAL, (nxdp+2)*(ny+2));
   }
   if(monogpu) {
   Device_ExecuteKernel(moveplusforcingconstructPBC_1_thermal)(pF1Source, pF1Dest);
    Device_Synchronize();
   }

   for(i=0; i<npop; i++) {
     Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
   }

   Device_ExecuteKernel(moveplusforcingconstructPBC_2_thermal)(pF1Source, pF1Dest);
    Device_Synchronize();

   for(i=0; i<npop; i++) {
     Device_SafeMemoryCopyDevice(pHostF1Source->p[i], pHostF1Dest->p[i], REAL, (nxdp+2)*(ny+2));
   }
    Device_Synchronize();

   Device_ExecuteKernel(moveplusforcingconstructPBC_3_thermal)(pF1Source, pF1Dest);
   Device_Synchronize();

   rhocomp(rho1,pF1Dest);

}


Device_QualKernel void AVERAGE_1(REAL *u1pre,REAL *v1pre,REAL *rho1pre,REAL *u2pre,REAL *v2pre,REAL *rho2pre,REAL *u1post,REAL *v1post,REAL *rho1post,REAL *u2post,REAL *v2post,REAL *rho2post,REAL *utot,REAL *vtot) {
  unsigned int idx0;
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
      set_i_j; i++; j++;
      idx0=j+(sNy+2)*i;

    utot[idx0]=(v(0.5)*((rho1pre[idx0]*u1pre[idx0]+rho2pre[idx0]*u2pre[idx0])+(rho1post[idx0]*u1post[idx0]+rho2post[idx0]*u2post[idx0])))/(rho1pre[idx0]+rho2pre[idx0]);

    vtot[idx0]=(v(0.5)*((rho1pre[idx0]*v1pre[idx0]+rho2pre[idx0]*v2pre[idx0])+(rho1post[idx0]*v1post[idx0]+rho2post[idx0]*v2post[idx0])))/(rho1pre[idx0]+rho2pre[idx0]);
  Device_ParallelEndRepeat
}

void AVERAGE(REAL *u1pre,REAL *v1pre,REAL *rho1pre,REAL *u2pre,REAL *v2pre,REAL *rho2pre,REAL *u1post,REAL *v1post,REAL *rho1post,REAL *u2post,REAL *v2post,REAL *rho2post,REAL *utot,REAL *vtot) {

  Device_ExecuteKernel(AVERAGE_1)(u1pre, v1pre, rho1pre, u2pre, v2pre, rho2pre, u1post, v1post, rho1post, u2post, v2post, rho2post, utot, vtot);
    Device_Synchronize();
}


Device_QualKernel void rhocomp_1(REAL *rho,pop_type *pF) {
  unsigned int idx1, k;
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;
    idx1=j+(sNy+2)*i;

    rho[idx1]=v(0.0);

    for(k=0;k<npop;k++){
      rho[idx1]=rho[idx1]+pF->p[k][idx1];
    }
  Device_ParallelEndRepeat

}

void rhocomp(REAL *rho,pop_type *pF) {
  Device_ExecuteKernel(rhocomp_1)(rho, pF);
    Device_Synchronize();
}

Device_QualKernel void hydrovar_1(REAL *u,REAL *v,REAL *rho,pop_type *pF,
                                      int *flag, REAL rhoobs) {
  REAL rhoi;
  REAL fl[npop];
  unsigned int k;
  unsigned int idx1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
      set_i_j; i++; j++;
      idx1=j+(sNy+2)*i;

      rho[idx1]=flag[idx1]?v(0.0):rhoobs;

      for(k=0;k<npop;k++){
        rho[idx1]=rho[idx1]+flag[idx1]*(pF->p[k][idx1]);
      }

      rhoi=v(1.0)/rho[idx1];

      for(k=0;k<npop;k++){
        fl[k]=pF->p[k][idx1]*rhoi;
      }

      u[idx1]=flag[idx1]*(fl[1]-fl[3]+fl[5]-fl[6]-fl[7]+fl[8]);
      v[idx1]=flag[idx1]*(fl[5]+fl[2]+fl[6]-fl[7]-fl[4]-fl[8]);
    }
  }
}

void hydrovar(REAL *u,REAL *v,REAL *rho,pop_type *pF, int *flag, REAL rhoobs) {

  Device_ExecuteKernel(hydrovar_1)(u, v, rho, pF, flag, rhoobs);
    Device_Synchronize();

}


void hydrovarA(REAL *u,REAL *v,REAL *rho,pop_type *pF, int *flag, REAL rhoobs) {

  Device_ExecuteKernelAS(hydrovar_1,1)(u, v, rho, pF, flag, rhoobs);

}


Device_QualKernel void thermalvar_1(REAL *rho,pop_type *pF, int *flag) {

  unsigned int k;
  unsigned int idx1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
      set_i_j; i++; j++;
      idx1=j+(sNy+2)*i;

      rho[idx1]=v(0.0);

      for(k=0;k<npop;k++){
        rho[idx1]=rho[idx1]+flag[idx1]*(pF->p[k][idx1]);
      }

    }
  }
}

void thermalvar(REAL *rho,pop_type *pF, int *flag) {

  Device_ExecuteKernel(thermalvar_1)(rho, pF, flag);
    Device_Synchronize();

}


void thermalvarA(REAL *rho,pop_type *pF, int *flag) {

  Device_ExecuteKernelAS(thermalvar_1,1)(rho, pF, flag);

}


void media(REAL *rho1,REAL *rho2, int *flag) {

  REAL rhoaveraged1,rhoaveraged2,volvoid;
  FILE *fout;

  rhoaveraged1=v(0.0);
  rhoaveraged2=v(0.0);

  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      if(flag[idx1]==1){rhoaveraged1+=rho1[idx1];}
    }
  }
  rhoaveraged1=rhoaveraged1/countflag;


  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      if(flag[idx1]==1){rhoaveraged2+=rho2[idx1];}
    }
  }
  rhoaveraged2=rhoaveraged2/countflag;


  volvoid=0.0;
  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      if(rho1[idx1]>0.5*(rhol+rhog)){volvoid=volvoid+1.;}
    }
  }

  volvoid=volvoid*innxny;


  fout=Fopen("Void.dat","w");

  fprintf(fout,"Mass Fraction of A into B = %e\nVolume of high density of A into total Volume = %e ",rhoaveraged1/(rhoaveraged1+rhoaveraged2),volvoid);

  fclose(fout);

  fout=Fopen("media.dat","w");

  fprintf(fout,"averaged bulk density 1 = %e \naveraged bulk density 2 = %e \ninitial averaged bulk density 1 = %e \ninitial averaged bulk density 2 = %e\nTotal = %e\nCenter Density A = %e\nCenter Density B = %e\nCount Flag = %e\n ",rhoaveraged1,rhoaveraged2,rhoin1,rhoin2,rhoin1+rhoin2,rho1[ny/2+(ny+2)*nx/2],rho2[ny/2+(ny+2)*nx/2],countflag);

  fclose(fout);

  printf("AVERAGES:  rho1 = %e, rho2 = %e \n",rhoaveraged1,rhoaveraged2);

}

void media_thermal(REAL *rho1, int *flag) {

  REAL rhoaveraged1,volvoid;
  FILE *fout;

  rhoaveraged1=v(0.0);

  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      if(flag[idx1]==1){rhoaveraged1+=rho1[idx1];}
    }
  }
  rhoaveraged1=rhoaveraged1/countflag;


  volvoid=0.0;
  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      if(rho1[idx1]>0.5*(rhol+rhog)){volvoid=volvoid+1.;}
    }}

  volvoid=volvoid*innxny;


  fout=Fopen("media_thermal.dat","w");

  fprintf(fout,"averaged bulk temperature = %e \naveraged  \ninitial averaged temperature = %e \nCenter temperature = %e\nCount Flag = %e\n ",rhoaveraged1,rhoin1,rho1[ny/2+(ny+2)*nx/2],countflag);

  fclose(fout);

  printf("AVERAGE TEMPERATURE = %e\n",rhoaveraged1);

}

float densitaTotaleMedia(REAL *rho1,REAL *rho2, int *flag) {

  REAL rhoaveraged;

  rhoaveraged=v(0.0);

  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      if(flag[idx1]==1){
	rhoaveraged += (rho1[idx1]+rho2[idx1]);
      }
    }
  }
  rhoaveraged = rhoaveraged/(nx*ny);

  printf("AVERAGE DENSITY = %e \n",rhoaveraged);

  return rhoaveraged;
}


Device_QualKernel void equili_1(REAL *u,REAL *v,REAL *rho,pop_type *pFeq) {
  REAL usq,vsq,sumsq,sumsq2,u2,v2,ui,vi,uv;

  unsigned int idx0;
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2Ny2)
    i = (index / (sNy+2)); j = (index - (i*(sNy+2)));
    idx0=j+(sNy+2)*i;

    usq=u[idx0]*u[idx0];
    vsq=v[idx0]*v[idx0];
    sumsq=(usq+vsq)/cs22;
    sumsq2=sumsq*(v(1.0)-cs2)/cs2;
    u2=usq/cssq;
    v2=vsq/cssq;
    ui=u[idx0]/cs2;
    vi=v[idx0]/cs2;
    uv=ui*vi;

    pFeq->p[0][idx0]=rho[idx0]*sRt0*(v(1.0)-sumsq);
    pFeq->p[1][idx0]=rho[idx0]*sRt1*(v(1.0)-sumsq+u2+ui);
    pFeq->p[2][idx0]=rho[idx0]*sRt1*(v(1.0)-sumsq+v2+vi);
    pFeq->p[3][idx0]=rho[idx0]*sRt1*(v(1.0)-sumsq+u2-ui);
    pFeq->p[4][idx0]=rho[idx0]*sRt1*(v(1.0)-sumsq+v2-vi);
    pFeq->p[5][idx0]=rho[idx0]*sRt2*(v(1.0)+sumsq2+ui+vi+uv);
    pFeq->p[6][idx0]=rho[idx0]*sRt2*(v(1.0)+sumsq2-ui+vi-uv);
    pFeq->p[7][idx0]=rho[idx0]*sRt2*(v(1.0)+sumsq2-ui-vi+uv);
    pFeq->p[8][idx0]=rho[idx0]*sRt2*(v(1.0)+sumsq2+ui-vi-uv);
  Device_ParallelEndRepeat

}

void equili(REAL *u,REAL *v,REAL *rho,pop_type *pFeq) {
  Device_ExecuteKernel(equili_1)(u, v, rho, pFeq);
    Device_Synchronize();
}

Device_QualKernel void equili_1_thermal(REAL *rho,pop_type *pFeq) {

  unsigned int idx0;
  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2Ny2)
    i = (index / (sNy+2)); j = (index - (i*(sNy+2)));
    idx0=j+(sNy+2)*i;

    pFeq->p[0][idx0]=rho[idx0]*sRt0;
    pFeq->p[1][idx0]=rho[idx0]*sRt1;
    pFeq->p[2][idx0]=rho[idx0]*sRt1;
    pFeq->p[3][idx0]=rho[idx0]*sRt1;
    pFeq->p[4][idx0]=rho[idx0]*sRt1;
    pFeq->p[5][idx0]=rho[idx0]*sRt2;
    pFeq->p[6][idx0]=rho[idx0]*sRt2;
    pFeq->p[7][idx0]=rho[idx0]*sRt2;
    pFeq->p[8][idx0]=rho[idx0]*sRt2;
  Device_ParallelEndRepeat

}

void equili_thermal(REAL *rho,pop_type *pFeq) {
  Device_ExecuteKernel(equili_1_thermal)(rho, pFeq);
    Device_Synchronize();
}


void mediatutto(pop_type f1,pop_type f2,REAL *psi1self,REAL *psi2self,REAL *rho1pre,REAL *rho2pre,REAL *utot,REAL *vtot,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *pxx,REAL *pxy11,REAL *pxy12,REAL *pxy22,REAL *pyy,REAL *pxxkin,REAL *pxykin,REAL *pyykin) {

  int plusx1,plusy1;
  int minusx1,minusy1;

  int plusx2,plusy2;
  int minusx2,minusy2;

  REAL uav;

  REAL Pxyavkin,Pxyav11,Pxyav12,Pxyav22;

  REAL valnow;

  FILE *fout1, *fout2;
  char filename[MAXFILENAME];

  REAL invRho2 = 1./(rho0*rho0);

  if(noutave>0 && (istep%noutave)==0) {
    snprintf(filename,sizeof(filename),"u_av.%d.dat",icount[AVEROUT]);
    fout1=Fopen(filename,"w");
    snprintf(filename,sizeof(filename),"u_av.dat");
    fout2=Fopen(filename,"w");

    for(j=1;j<=ny;j++){
      uav=v(0.0);
      for(i=1;i<=nx;i++){
        idx0=j+(ny+2)*i;
        uav+=utot[idx0];
      }
      uav=uav*innx;
      fprintf(fout1,"%d %e \n",j,uav);
      fprintf(fout2,"%d %e \n",j,uav);
    }
    fclose(fout1);
    fclose(fout2);
  }

  if((noutvelo>0 && (istep%noutvelo)==0)&&(dooutvtk==0)) {
    snprintf(filename,sizeof(filename),"veloconf.%d.dat",icount[VELOCITY]);
    fout1=Fopen(filename,"w");
    snprintf(filename,sizeof(filename),"veloconf.dat");
    fout2=Fopen(filename,"w");
    for(i=1;i<=nx;i++){
      for(j=1;j<=ny;j++){
        idx1=j+(ny+2)*i;
        fprintf(fout1,"%d %d %e %e\n",i,j,utot[idx1],vtot[idx1]);
        fprintf(fout2,"%d %d %e %e\n",i,j,utot[idx1],vtot[idx1]);
      }
      fprintf(fout1,"\n");
      fprintf(fout2,"\n");
    }
    fclose(fout1);
    fclose(fout2);
    icount[VELOCITY]++;
  }


  if(((noutvelo>0)&&(istep%noutvelo)==0)&&(dooutvtk==1)) {
    snprintf(filename,sizeof(filename),"veloconf.%d.vtk",icount[VTK]);
    fout1 = fopen(filename, "w");
    fprintf(fout1,"# vtk DataFile Version 2.0\n");
    fprintf(fout1,"CAMPO\n");
    fprintf(fout1,"ASCII\n");
    fprintf(fout1,"DATASET STRUCTURED_POINTS\n");
    fprintf(fout1,"DIMENSIONS %d %d %d\n",nx,ny,1);
    fprintf(fout1,"ORIGIN %d %d %d\n",0,0,0);
    fprintf(fout1,"SPACING 1 1 1\n");
    fprintf(fout1,"POINT_DATA %d\n",nx*ny*1);
    fprintf(fout1,"VECTORS velocity double \n");

    for (j=1;j<=ny;j++){
      for (i=1;i<=nx;i++){
        idx0=j+(ny+2)*i;
        fprintf(fout1,"%f %f %f\n",utot[idx0],vtot[idx0],0.0);
      }
    }
    fclose(fout1);
  }




  if(((nouttens>0) && (istep%nouttens)==0) || (noutave>0 && (istep%noutave)==0)) {

    for(i=1;i<=nx-1;i++){
      for(j=1;j<=ny-1;j++){

        plusx1=i+1;
        minusx1=i-1;
        plusy1=j+1;
        minusy1=j-1;

        plusx2=i+2;
        minusx2=i-2;
        plusy2=j+2;
        minusy2=j-2;

        idx0=j+(ny+2)*i;

        idx1=j+(ny+2)*plusx1;
        idx2=plusy1+(ny+2)*i;
        idx3=j+(ny+2)*minusx1;
        idx4=minusy1+(ny+2)*i;

        idx5=plusy1+(ny+2)*plusx1;
        idx6=plusy1+(ny+2)*minusx1;
        idx7=minusy1+(ny+2)*minusx1;
        idx8=minusy1+(ny+2)*plusx1;

        idx9=j+(ny+2)*plusx2;
        idx10=plusy2+(ny+2)*i;
        idx11=j+(ny+2)*minusx2;
        idx12=minusy2+(ny+2)*i;

        idx13=plusy1+(ny+2)*plusx2;
        idx14=plusy2+(ny+2)*plusx1;
        idx15=plusy2+(ny+2)*minusx1;
        idx16=plusy1+(ny+2)*minusx2;

        idx17=minusy1+(ny+2)*minusx2;
        idx18=minusy2+(ny+2)*minusx1;

        idx19=minusy2+(ny+2)*plusx1;
        idx20=minusy1+(ny+2)*plusx2;

        idx21=plusy2+(ny+2)*plusx2;
        idx22=plusy2+(ny+2)*minusx2;
        idx23=minusy2+(ny+2)*minusx2;
        idx24=minusy2+(ny+2)*plusx2;


        pxykin[idx0]=0.0;
        pxxkin[idx0]=0.0;
        pyykin[idx0]=0.0;

        for(k=0;k<npop;k++){
          pxykin[idx0]+=(f1.p[k][idx0]+f2.p[k][idx0])*cx[k]*cy[k];
          pxxkin[idx0]+=(f1.p[k][idx0]+f2.p[k][idx0])*cx[k]*cx[k];
          pyykin[idx0]+=(f1.p[k][idx0]+f2.p[k][idx0])*cy[k]*cy[k];
        }


	
        pxykin[idx0]=+(f1.p[5][idx7]+f2.p[5][idx7])-(f1.p[6][idx8]+f2.p[6][idx8])+(f1.p[7][idx0]+f2.p[7][idx0])-(f1.p[8][idx0]+f2.p[8][idx0]);

        pxy12[idx0]=+0.5*invRho2*g12*rho1pre[idx0]*(ww[1]*rho2pre[idx1]*cx[1]*cy[1]+ww[2]*rho2pre[idx2]*cx[2]*cy[2]+ww[3]*rho2pre[idx3]*cx[3]*cy[3]+ww[4]*rho2pre[idx4]*cx[4]*cy[4]+ww[5]*rho2pre[idx5]*cx[5]*cy[5]+ww[6]*rho2pre[idx6]*cx[6]*cy[6]+ww[7]*rho2pre[idx7]*cx[7]*cy[7]+ww[8]*rho2pre[idx8]*cx[8]*cy[8])

          +0.5*invRho2*g12*rho2pre[idx0]*(ww[1]*rho1pre[idx1]*cx[1]*cy[1]+ww[2]*rho1pre[idx2]*cx[2]*cy[2]+ww[3]*rho1pre[idx3]*cx[3]*cy[3]+ww[4]*rho1pre[idx4]*cx[4]*cy[4]+ww[5]*rho1pre[idx5]*cx[5]*cy[5]+ww[6]*rho1pre[idx6]*cx[6]*cy[6]+ww[7]*rho1pre[idx7]*cx[7]*cy[7]+ww[8]*rho1pre[idx8]*cx[8]*cy[8]);


        pxy11[idx0]=+0.5*G1a*psi1self[idx0]*(ww[1]*psi1self[idx1]*cx[1]*cy[1]+ww[2]*psi1self[idx2]*cx[2]*cy[2]+ww[3]*psi1self[idx3]*cx[3]*cy[3]+ww[4]*psi1self[idx4]*cx[4]*cy[4]+ww[5]*psi1self[idx5]*cx[5]*cy[5]+ww[6]*psi1self[idx6]*cx[6]*cy[6]+ww[7]*psi1self[idx7]*cx[7]*cy[7]+ww[8]*psi1self[idx8]*cx[8]*cy[8])

          +0.5*G1r*psi1self[idx0]*(p[1]*psi1self[idx1]*cx[1]*cy[1]+p[2]*psi1self[idx2]*cx[2]*cy[2]+p[3]*psi1self[idx3]*cx[3]*cy[3]+p[4]*psi1self[idx4]*cx[4]*cy[4]+p[5]*psi1self[idx5]*cx[5]*cy[5]+p[6]*psi1self[idx6]*cx[6]*cy[6]+p[7]*psi1self[idx7]*cx[7]*cy[7]+p[8]*psi1self[idx8]*cx[8]*cy[8])

          +0.25*G1r*psi1self[idx0]*(p[9]*psi1self[idx9]*cx[9]*cy[9]+p[10]*psi1self[idx10]*cx[10]*cy[10]+p[11]*psi1self[idx11]*cx[11]*cy[11]+p[12]*psi1self[idx12]*cx[12]*cy[12]+p[21]*psi1self[idx21]*cx[21]*cy[21]+p[22]*psi1self[idx22]*cx[22]*cy[22]+p[23]*psi1self[idx23]*cx[23]*cy[23]+p[24]*psi1self[idx24]*cx[24]*cy[24])

          +0.25*G1r*(p[9]*psi1self[idx1]*psi1self[idx3]*cx[9]*cy[9]+p[10]*psi1self[idx2]*psi1self[idx4]*cx[10]*cy[10]+p[11]*psi1self[idx1]*psi1self[idx3]*cx[11]*cy[11]+p[12]*psi1self[idx4]*psi1self[idx2]*cx[12]*cy[12]+p[21]*psi1self[idx5]*psi1self[idx7]*cx[21]*cy[21]+p[22]*psi1self[idx6]*psi1self[idx8]*cx[22]*cy[22]+p[23]*psi1self[idx7]*psi1self[idx5]*cx[23]*cy[23]+p[24]*psi1self[idx8]*psi1self[idx6]*cx[24]*cy[24])

          +0.25*G1r*psi1self[idx0]*(p[13]*psi1self[idx13]*cx[13]*cy[13]+p[14]*psi1self[idx14]*cx[14]*cy[14]+p[15]*psi1self[idx15]*cx[15]*cy[15]+p[16]*psi1self[idx16]*cx[16]*cy[16]+p[17]*psi1self[idx17]*cx[17]*cy[17]+p[18]*psi1self[idx18]*cx[18]*cy[18]+p[19]*psi1self[idx19]*cx[19]*cy[19]+p[20]*psi1self[idx20]*cx[20]*cy[20])
          +0.25*G1r*(p[13]*(psi1self[idx5]*psi1self[idx3]+psi1self[idx1]*psi1self[idx7])*cx[13]*cy[13]+p[14]*(psi1self[idx5]*psi1self[idx4]+psi1self[idx2]*psi1self[idx7])*cx[14]*cy[14]+p[15]*(psi1self[idx2]*psi1self[idx8]+psi1self[idx6]*psi1self[idx4])*cx[15]*cy[15]+p[16]*(psi1self[idx6]*psi1self[idx1]+psi1self[idx8]*psi1self[idx3])*cx[16]*cy[16]);



        pxy22[idx0]=+0.5*G2a*psi2self[idx0]*(ww[1]*psi2self[idx1]*cx[1]*cy[1]+ww[2]*psi2self[idx2]*cx[2]*cy[2]+ww[3]*psi2self[idx3]*cx[3]*cy[3]+ww[4]*psi2self[idx4]*cx[4]*cy[4]+ww[5]*psi2self[idx5]*cx[5]*cy[5]+ww[6]*psi2self[idx6]*cx[6]*cy[6]+ww[7]*psi2self[idx7]*cx[7]*cy[7]+ww[8]*psi2self[idx8]*cx[8]*cy[8])

          +0.5*G2r*psi2self[idx0]*(p[1]*psi2self[idx1]*cx[1]*cy[1]+p[2]*psi2self[idx2]*cx[2]*cy[2]+p[3]*psi2self[idx3]*cx[3]*cy[3]+p[4]*psi2self[idx4]*cx[4]*cy[4]+p[5]*psi2self[idx5]*cx[5]*cy[5]+p[6]*psi2self[idx6]*cx[6]*cy[6]+p[7]*psi2self[idx7]*cx[7]*cy[7]+p[8]*psi2self[idx8]*cx[8]*cy[8])

          +0.25*G2r*psi2self[idx0]*(p[9]*psi2self[idx9]*cx[9]*cy[9]+p[10]*psi2self[idx10]*cx[10]*cy[10]+p[11]*psi2self[idx11]*cx[11]*cy[11]+p[12]*psi2self[idx12]*cx[12]*cy[12]+p[21]*psi2self[idx21]*cx[21]*cy[21]+p[22]*psi2self[idx22]*cx[22]*cy[22]+p[23]*psi2self[idx23]*cx[23]*cy[23]+p[24]*psi2self[idx24]*cx[24]*cy[24])

          +0.25*G2r*(p[9]*psi2self[idx1]*psi2self[idx3]*cx[9]*cy[9]+p[10]*psi2self[idx2]*psi2self[idx4]*cx[10]*cy[10]+p[11]*psi2self[idx1]*psi2self[idx3]*cx[11]*cy[11]+p[12]*psi2self[idx4]*psi2self[idx2]*cx[12]*cy[12]+p[21]*psi2self[idx5]*psi2self[idx7]*cx[21]*cy[21]+p[22]*psi2self[idx6]*psi2self[idx8]*cx[22]*cy[22]+p[23]*psi2self[idx7]*psi2self[idx5]*cx[23]*cy[23]+p[24]*psi2self[idx8]*psi2self[idx6]*cx[24]*cy[24])

          +0.25*G2r*psi2self[idx0]*(p[13]*psi2self[idx13]*cx[13]*cy[13]+p[14]*psi2self[idx14]*cx[14]*cy[14]+p[15]*psi2self[idx15]*cx[15]*cy[15]+p[16]*psi2self[idx16]*cx[16]*cy[16]+p[17]*psi2self[idx17]*cx[17]*cy[17]+p[18]*psi2self[idx18]*cx[18]*cy[18]+p[19]*psi2self[idx19]*cx[19]*cy[19]+p[20]*psi2self[idx20]*cx[20]*cy[20])
          +0.25*G2r*(p[13]*(psi2self[idx5]*psi2self[idx3]+psi2self[idx1]*psi2self[idx7])*cx[13]*cy[13]+p[14]*(psi2self[idx5]*psi2self[idx4]+psi2self[idx2]*psi2self[idx7])*cx[14]*cy[14]+p[15]*(psi2self[idx2]*psi2self[idx8]+psi2self[idx6]*psi2self[idx4])*cx[15]*cy[15]+p[16]*(psi2self[idx6]*psi2self[idx1]+psi2self[idx8]*psi2self[idx3])*cx[16]*cy[16]);


      }
    }


    if((noutave>0 && (istep%noutave)==0)) {
    snprintf(filename,sizeof(filename),"Pxy_av.%d.dat",icount[AVEROUT]);
    icount[AVEROUT]++;
    fout1=Fopen(filename,"w");
    snprintf(filename,sizeof(filename),"Pxy_av.dat");
    fout2=Fopen(filename,"w");

    for(j=2;j<=ny-1;j++){

      Pxyavkin=0.0;

      Pxyav11=0.0;
      Pxyav12=0.0;
      Pxyav22=0.0;

      for(i=2;i<=nx-1;i++){

        plusx1=i+1;
        minusx1=i-1;
        plusy1=j+1;
        minusy1=j-1;

        plusx2=i+2;
        minusx2=i-2;
        plusy2=j+2;
        minusy2=j-2;

        idx0=j+(ny+2)*i;

        idx1=j+(ny+2)*plusx1;
        idx2=plusy1+(ny+2)*i;
        idx3=j+(ny+2)*minusx1;
        idx4=minusy1+(ny+2)*i;

        idx5=plusy1+(ny+2)*plusx1;
        idx6=plusy1+(ny+2)*minusx1;
        idx7=minusy1+(ny+2)*minusx1;
        idx8=minusy1+(ny+2)*plusx1;

        Pxyavkin+=pxykin[idx0];

        Pxyav11+=pxy11[idx0];
        Pxyav12+=pxy12[idx0];
        Pxyav22+=pxy22[idx0];

      }

      Pxyavkin=Pxyavkin*innxm2;

      Pxyav11=Pxyav11*innxm2;
      Pxyav12=Pxyav12*innxm2;
      Pxyav22=Pxyav22*innxm2;


      fprintf(fout1,"%d %e %e %e %e \n",j,Pxyavkin,Pxyav11,Pxyav12,Pxyav22);
      fprintf(fout2,"%d %e %e %e %e \n",j,Pxyavkin,Pxyav11,Pxyav12,Pxyav22);}


    fclose(fout1);
    fclose(fout2);
    }



  }

  if((nouttens > 0 && (istep%nouttens)==0)&&(dooutvtk==0)) {
    snprintf(filename,sizeof(filename),"Ptot_xy.%d.dat",icount[TENSOR]);
    fout1=Fopen(filename,"w");
    snprintf(filename,sizeof(filename),"Ptot_xy.dat");
    fout2=Fopen(filename,"w");

    for(i=1;i<=nx;i++){
      for(j=2;j<=ny-1;j++){
        idx1=j+(ny+2)*i;
        fprintf(fout1,"%d %d %e %e %e %e \n",i,j,pxykin[idx1],pxy11[idx1],pxy12[idx1],pxy22[idx1]);
        fprintf(fout2,"%d %d %e %e %e %e \n",i,j,pxykin[idx1],pxy11[idx1],pxy12[idx1],pxy22[idx1]);}
      fprintf(fout1,"\n");
      fprintf(fout2,"\n");
    }
    fclose(fout1);

    icount[TENSOR]++;
  }

  if((nouttens>0 && (istep%nouttens)==0)&&(dooutvtk==1)){
        snprintf(filename,sizeof(filename),"Pxy.%d.vtk",icount[VTK]);
        fout1 = fopen(filename, "w");
        fprintf(fout1,"# vtk DataFile Version 2.0\n");
        fprintf(fout1,"CAMPO\n");
        fprintf(fout1,"ASCII\n");
        fprintf(fout1,"DATASET STRUCTURED_POINTS\n");
        fprintf(fout1,"DIMENSIONS %d %d %d\n",nx,ny,1);
        fprintf(fout1,"ORIGIN %d %d %d\n",0,0,0);
        fprintf(fout1,"SPACING 1 1 1\n");
        fprintf(fout1,"POINT_DATA %d\n",nx*ny*1);
        fprintf(fout1,"SCALARS Pxy double 1\n");
        fprintf(fout1,"LOOKUP_TABLE default\n");
        for(j=1;j<=ny;j++){
          for(i=1;i<=nx;i++){
            idx0=j+(ny+2)*i;
            valnow=pxykin[idx0]+pxy11[idx0]+pxy12[idx0]+pxy22[idx0];
            fprintf(fout1,"%e\n",valnow);
          }
        }
        fclose(fout1);

  }



}

void computeEnergy(REAL *utot, REAL *vtot) {

  REAL energy = 0.;
  char filename[MAXFILENAME];

  snprintf(filename,sizeof(filename),"timeEnergy.dat");
  FILE *fout=Fopen(filename,"a+");

  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;	
      energy += (utot[idx1]*utot[idx1]) + (vtot[idx1]*vtot[idx1]);
    }
  }
  energy /= (nx*ny);
  fprintf(fout,"%d %e\n",istep,energy);
  
  energy = 0.;
  
  fflush(fout);
  fclose(fout);

}

void computeptensor(pop_type f1,pop_type f2,REAL *psi1self,REAL *psi2self,REAL *rho1pre,REAL *rho2pre,REAL *pxx,REAL *pxy,REAL *pyy,REAL *pxxkin,REAL *pxykin,REAL *pyykin) {

  int plusx1,plusy1;
  int minusx1,minusy1;

  int plusx2,plusy2;
  int minusx2,minusy2;

  REAL invRho2 = 1./(rho0*rho0);

  for(i=1;i<=nx-1;i++){
    for(j=1;j<=ny-1;j++){

      plusx1=i+1;
      minusx1=i-1;
      plusy1=j+1;
      minusy1=j-1;

      plusx2=i+2;
      minusx2=i-2;
      plusy2=j+2;
      minusy2=j-2;

      idx0=j+(ny+2)*i;

      idx1=j+(ny+2)*plusx1;
      idx2=plusy1+(ny+2)*i;
      idx3=j+(ny+2)*minusx1;
      idx4=minusy1+(ny+2)*i;

      idx5=plusy1+(ny+2)*plusx1;
      idx6=plusy1+(ny+2)*minusx1;
      idx7=minusy1+(ny+2)*minusx1;
      idx8=minusy1+(ny+2)*plusx1;

      idx9=j+(ny+2)*plusx2;
      idx10=plusy2+(ny+2)*i;
      idx11=j+(ny+2)*minusx2;
      idx12=minusy2+(ny+2)*i;

      idx13=plusy1+(ny+2)*plusx2;
      idx14=plusy2+(ny+2)*plusx1;
      idx15=plusy2+(ny+2)*minusx1;
      idx16=plusy1+(ny+2)*minusx2;

      idx17=minusy1+(ny+2)*minusx2;
      idx18=minusy2+(ny+2)*minusx1;
      idx19=minusy2+(ny+2)*plusx1;
      idx20=minusy1+(ny+2)*plusx2;

      idx21=plusy2+(ny+2)*plusx2;
      idx22=plusy2+(ny+2)*minusx2;
      idx23=minusy2+(ny+2)*minusx2;
      idx24=minusy2+(ny+2)*plusx2;


      pxykin[idx0]=v(0.0);
      pxxkin[idx0]=v(0.0);
      pyykin[idx0]=v(0.0);

      for(k=0;k<npop;k++){
        pxykin[idx0]+=(f1.p[k][idx0]+f2.p[k][idx0])*cx[k]*cy[k];
        pxxkin[idx0]+=(f1.p[k][idx0]+f2.p[k][idx0])*cx[k]*cx[k];
        pyykin[idx0]+=(f1.p[k][idx0]+f2.p[k][idx0])*cy[k]*cy[k];
      }


      pxykin[idx0]=+(f1.p[5][idx7]+f2.p[5][idx7])-(f1.p[6][idx8]+f2.p[6][idx8])+(f1.p[7][idx0]+f2.p[7][idx0])-(f1.p[8][idx0]+f2.p[8][idx0]);

      pxx[idx0]=+v(0.5)*invRho2*g12*rho1pre[idx0]*(ww[1]*rho2pre[idx1]*cx[1]*cx[1]+ww[2]*rho2pre[idx2]*cx[2]*cx[2]+ww[3]*rho2pre[idx3]*cx[3]*cx[3]+ww[4]*rho2pre[idx4]*cx[4]*cx[4]+ww[5]*rho2pre[idx5]*cx[5]*cx[5]+ww[6]*rho2pre[idx6]*cx[6]*cx[6]+ww[7]*rho2pre[idx7]*cx[7]*cx[7]+ww[8]*rho2pre[idx8]*cx[8]*cx[8])

      +v(0.5)*invRho2*g12*rho2pre[idx0]*(ww[1]*rho1pre[idx1]*cx[1]*cx[1]+ww[2]*rho1pre[idx2]*cx[2]*cx[2]+ww[3]*rho1pre[idx3]*cx[3]*cx[3]+ww[4]*rho1pre[idx4]*cx[4]*cx[4]+ww[5]*rho1pre[idx5]*cx[5]*cx[5]+ww[6]*rho1pre[idx6]*cx[6]*cx[6]+ww[7]*rho1pre[idx7]*cx[7]*cx[7]+ww[8]*rho1pre[idx8]*cx[8]*cx[8])

      +v(0.5)*G1a*psi1self[idx0]*(ww[1]*psi1self[idx1]*cx[1]*cx[1]+ww[2]*psi1self[idx2]*cx[2]*cx[2]+ww[3]*psi1self[idx3]*cx[3]*cx[3]+ww[4]*psi1self[idx4]*cx[4]*cx[4]+ww[5]*psi1self[idx5]*cx[5]*cx[5]+ww[6]*psi1self[idx6]*cx[6]*cx[6]+ww[7]*psi1self[idx7]*cx[7]*cx[7]+ww[8]*psi1self[idx8]*cx[8]*cx[8])

      +v(0.5)*G2a*psi2self[idx0]*(ww[1]*psi2self[idx1]*cx[1]*cx[1]+ww[2]*psi2self[idx2]*cx[2]*cx[2]+ww[3]*psi2self[idx3]*cx[3]*cx[3]+ww[4]*psi2self[idx4]*cx[4]*cx[4]+ww[5]*psi2self[idx5]*cx[5]*cx[5]+ww[6]*psi2self[idx6]*cx[6]*cx[6]+ww[7]*psi2self[idx7]*cx[7]*cx[7]+ww[8]*psi2self[idx8]*cx[8]*cx[8])

      +v(0.5)*G1r*psi1self[idx0]*(p[1]*psi1self[idx1]*cx[1]*cx[1]+p[2]*psi1self[idx2]*cx[2]*cx[2]+p[3]*psi1self[idx3]*cx[3]*cx[3]+p[4]*psi1self[idx4]*cx[4]*cx[4]+p[5]*psi1self[idx5]*cx[5]*cx[5]+p[6]*psi1self[idx6]*cx[6]*cx[6]+p[7]*psi1self[idx7]*cx[7]*cx[7]+p[8]*psi1self[idx8]*cx[8]*cx[8])

      +v(0.25)*G1r*psi1self[idx0]*(p[9]*psi1self[idx9]*cx[9]*cx[9]+p[10]*psi1self[idx10]*cx[10]*cx[10]+p[11]*psi1self[idx11]*cx[11]*cx[11]+p[12]*psi1self[idx12]*cx[12]*cx[12]+p[21]*psi1self[idx21]*cx[21]*cx[21]+p[22]*psi1self[idx22]*cx[22]*cx[22]+p[23]*psi1self[idx23]*cx[23]*cx[23]+p[24]*psi1self[idx24]*cx[24]*cx[24])

      +v(0.25)*G1r*(p[9]*psi1self[idx1]*psi1self[idx3]*cx[9]*cx[9]+p[10]*psi1self[idx2]*psi1self[idx4]*cx[10]*cx[10]+p[11]*psi1self[idx1]*psi1self[idx3]*cx[11]*cx[11]+p[12]*psi1self[idx4]*psi1self[idx2]*cx[12]*cx[12]+p[21]*psi1self[idx5]*psi1self[idx7]*cx[21]*cx[21]+p[22]*psi1self[idx6]*psi1self[idx8]*cx[22]*cx[22]+p[23]*psi1self[idx7]*psi1self[idx5]*cx[23]*cx[23]+p[24]*psi1self[idx8]*psi1self[idx6]*cx[24]*cx[24])

      +v(0.25)*G1r*psi1self[idx0]*(p[13]*psi1self[idx13]*cx[13]*cx[13]+p[14]*psi1self[idx14]*cx[14]*cx[14]+p[15]*psi1self[idx15]*cx[15]*cx[15]+p[16]*psi1self[idx16]*cx[16]*cx[16]+p[17]*psi1self[idx17]*cx[17]*cx[17]+p[18]*psi1self[idx18]*cx[18]*cx[18]+p[19]*psi1self[idx19]*cx[19]*cx[19]+p[20]*psi1self[idx20]*cx[20]*cx[20])

      +v(0.25)*G1r*(p[13]*(psi1self[idx5]*psi1self[idx3]+psi1self[idx1]*psi1self[idx7])*cx[13]*cx[13]+p[14]*(psi1self[idx5]*psi1self[idx4]+psi1self[idx2]*psi1self[idx7])*cx[14]*cx[14]+p[15]*(psi1self[idx2]*psi1self[idx8]+psi1self[idx6]*psi1self[idx4])*cx[15]*cx[15]+p[16]*(psi1self[idx6]*psi1self[idx1]+psi1self[idx8]*psi1self[idx3])*cx[16]*cx[16])


      +v(0.5)*G2r*psi2self[idx0]*(p[1]*psi2self[idx1]*cx[1]*cx[1]+p[2]*psi2self[idx2]*cx[2]*cx[2]+p[3]*psi2self[idx3]*cx[3]*cx[3]+p[4]*psi2self[idx4]*cx[4]*cx[4]+p[5]*psi2self[idx5]*cx[5]*cx[5]+p[6]*psi2self[idx6]*cx[6]*cx[6]+p[7]*psi2self[idx7]*cx[7]*cx[7]+p[8]*psi2self[idx8]*cx[8]*cx[8])

      +v(0.25)*G2r*psi2self[idx0]*(p[9]*psi2self[idx9]*cx[9]*cx[9]+p[10]*psi2self[idx10]*cx[10]*cx[10]+p[11]*psi2self[idx11]*cx[11]*cx[11]+p[12]*psi2self[idx12]*cx[12]*cx[12]+p[21]*psi2self[idx21]*cx[21]*cx[21]+p[22]*psi2self[idx22]*cx[22]*cx[22]+p[23]*psi2self[idx23]*cx[23]*cx[23]+p[24]*psi2self[idx24]*cx[24]*cx[24])

      +v(0.25)*G2r*(p[9]*psi2self[idx1]*psi2self[idx3]*cx[9]*cx[9]+p[10]*psi2self[idx2]*psi2self[idx4]*cx[10]*cx[10]+p[11]*psi2self[idx1]*psi2self[idx3]*cx[11]*cx[11]+p[12]*psi2self[idx4]*psi2self[idx2]*cx[12]*cx[12]+p[21]*psi2self[idx5]*psi2self[idx7]*cx[21]*cx[21]+p[22]*psi2self[idx6]*psi2self[idx8]*cx[22]*cx[22]+p[23]*psi2self[idx7]*psi2self[idx5]*cx[23]*cx[23]+p[24]*psi2self[idx8]*psi2self[idx6]*cx[24]*cx[24])

      +v(0.25)*G2r*psi2self[idx0]*(p[13]*psi2self[idx13]*cx[13]*cx[13]+p[14]*psi2self[idx14]*cx[14]*cx[14]+p[15]*psi2self[idx15]*cx[15]*cx[15]+p[16]*psi2self[idx16]*cx[16]*cx[16]+p[17]*psi2self[idx17]*cx[17]*cx[17]+p[18]*psi2self[idx18]*cx[18]*cx[18]+p[19]*psi2self[idx19]*cx[19]*cx[19]+p[20]*psi2self[idx20]*cx[20]*cx[20])

      +v(0.25)*G2r*(p[13]*(psi2self[idx5]*psi2self[idx3]+psi2self[idx1]*psi2self[idx7])*cx[13]*cx[13]+p[14]*(psi2self[idx5]*psi2self[idx4]+psi2self[idx2]*psi2self[idx7])*cx[14]*cx[14]+p[15]*(psi2self[idx2]*psi2self[idx8]+psi2self[idx6]*psi2self[idx4])*cx[15]*cx[15]+p[16]*(psi2self[idx6]*psi2self[idx1]+psi2self[idx8]*psi2self[idx3])*cx[16]*cx[16]);


      pxy[idx0]=+v(0.5)*invRho2*g12*rho1pre[idx0]*(ww[1]*rho2pre[idx1]*cx[1]*cy[1]+ww[2]*rho2pre[idx2]*cx[2]*cy[2]+ww[3]*rho2pre[idx3]*cx[3]*cy[3]+ww[4]*rho2pre[idx4]*cx[4]*cy[4]+ww[5]*rho2pre[idx5]*cx[5]*cy[5]+ww[6]*rho2pre[idx6]*cx[6]*cy[6]+ww[7]*rho2pre[idx7]*cx[7]*cy[7]+ww[8]*rho2pre[idx8]*cx[8]*cy[8])

      +v(0.5)*invRho2*g12*rho2pre[idx0]*(ww[1]*rho1pre[idx1]*cx[1]*cy[1]+ww[2]*rho1pre[idx2]*cx[2]*cy[2]+ww[3]*rho1pre[idx3]*cx[3]*cy[3]+ww[4]*rho1pre[idx4]*cx[4]*cy[4]+ww[5]*rho1pre[idx5]*cx[5]*cy[5]+ww[6]*rho1pre[idx6]*cx[6]*cy[6]+ww[7]*rho1pre[idx7]*cx[7]*cy[7]+ww[8]*rho1pre[idx8]*cx[8]*cy[8])

      +v(0.5)*G1a*psi1self[idx0]*(ww[1]*psi1self[idx1]*cx[1]*cy[1]+ww[2]*psi1self[idx2]*cx[2]*cy[2]+ww[3]*psi1self[idx3]*cx[3]*cy[3]+ww[4]*psi1self[idx4]*cx[4]*cy[4]+ww[5]*psi1self[idx5]*cx[5]*cy[5]+ww[6]*psi1self[idx6]*cx[6]*cy[6]+ww[7]*psi1self[idx7]*cx[7]*cy[7]+ww[8]*psi1self[idx8]*cx[8]*cy[8])

      +v(0.5)*G2a*psi2self[idx0]*(ww[1]*psi2self[idx1]*cx[1]*cy[1]+ww[2]*psi2self[idx2]*cx[2]*cy[2]+ww[3]*psi2self[idx3]*cx[3]*cy[3]+ww[4]*psi2self[idx4]*cx[4]*cy[4]+ww[5]*psi2self[idx5]*cx[5]*cy[5]+ww[6]*psi2self[idx6]*cx[6]*cy[6]+ww[7]*psi2self[idx7]*cx[7]*cy[7]+ww[8]*psi2self[idx8]*cx[8]*cy[8])

      +v(0.5)*G1r*psi1self[idx0]*(p[1]*psi1self[idx1]*cx[1]*cy[1]+p[2]*psi1self[idx2]*cx[2]*cy[2]+p[3]*psi1self[idx3]*cx[3]*cy[3]+p[4]*psi1self[idx4]*cx[4]*cy[4]+p[5]*psi1self[idx5]*cx[5]*cy[5]+p[6]*psi1self[idx6]*cx[6]*cy[6]+p[7]*psi1self[idx7]*cx[7]*cy[7]+p[8]*psi1self[idx8]*cx[8]*cy[8])

      +v(0.25)*G1r*psi1self[idx0]*(p[9]*psi1self[idx9]*cx[9]*cy[9]+p[10]*psi1self[idx10]*cx[10]*cy[10]+p[11]*psi1self[idx11]*cx[11]*cy[11]+p[12]*psi1self[idx12]*cx[12]*cy[12]+p[21]*psi1self[idx21]*cx[21]*cy[21]+p[22]*psi1self[idx22]*cx[22]*cy[22]+p[23]*psi1self[idx23]*cx[23]*cy[23]+p[24]*psi1self[idx24]*cx[24]*cy[24])

      +v(0.25)*G1r*(p[9]*psi1self[idx1]*psi1self[idx3]*cx[9]*cy[9]+p[10]*psi1self[idx2]*psi1self[idx4]*cx[10]*cy[10]+p[11]*psi1self[idx1]*psi1self[idx3]*cx[11]*cy[11]+p[12]*psi1self[idx4]*psi1self[idx2]*cx[12]*cy[12]+p[21]*psi1self[idx5]*psi1self[idx7]*cx[21]*cy[21]+p[22]*psi1self[idx6]*psi1self[idx8]*cx[22]*cy[22]+p[23]*psi1self[idx7]*psi1self[idx5]*cx[23]*cy[23]+p[24]*psi1self[idx8]*psi1self[idx6]*cx[24]*cy[24])

      +v(0.25)*G1r*psi1self[idx0]*(p[13]*psi1self[idx13]*cx[13]*cy[13]+p[14]*psi1self[idx14]*cx[14]*cy[14]+p[15]*psi1self[idx15]*cx[15]*cy[15]+p[16]*psi1self[idx16]*cx[16]*cy[16]+p[17]*psi1self[idx17]*cx[17]*cy[17]+p[18]*psi1self[idx18]*cx[18]*cy[18]+p[19]*psi1self[idx19]*cx[19]*cy[19]+p[20]*psi1self[idx20]*cx[20]*cy[20])

      +v(0.25)*G1r*(p[13]*(psi1self[idx5]*psi1self[idx3]+psi1self[idx1]*psi1self[idx7])*cx[13]*cy[13]+p[14]*(psi1self[idx5]*psi1self[idx4]+psi1self[idx2]*psi1self[idx7])*cx[14]*cy[14]+p[15]*(psi1self[idx2]*psi1self[idx8]+psi1self[idx6]*psi1self[idx4])*cx[15]*cy[15]+p[16]*(psi1self[idx6]*psi1self[idx1]+psi1self[idx8]*psi1self[idx3])*cx[16]*cy[16])


      +v(0.5)*G2r*psi2self[idx0]*(p[1]*psi2self[idx1]*cx[1]*cy[1]+p[2]*psi2self[idx2]*cx[2]*cy[2]+p[3]*psi2self[idx3]*cx[3]*cy[3]+p[4]*psi2self[idx4]*cx[4]*cy[4]+p[5]*psi2self[idx5]*cx[5]*cy[5]+p[6]*psi2self[idx6]*cx[6]*cy[6]+p[7]*psi2self[idx7]*cx[7]*cy[7]+p[8]*psi2self[idx8]*cx[8]*cy[8])

      +v(0.25)*G2r*psi2self[idx0]*(p[9]*psi2self[idx9]*cx[9]*cy[9]+p[10]*psi2self[idx10]*cx[10]*cy[10]+p[11]*psi2self[idx11]*cx[11]*cy[11]+p[12]*psi2self[idx12]*cx[12]*cy[12]+p[21]*psi2self[idx21]*cx[21]*cy[21]+p[22]*psi2self[idx22]*cx[22]*cy[22]+p[23]*psi2self[idx23]*cx[23]*cy[23]+p[24]*psi2self[idx24]*cx[24]*cy[24])

      +v(0.25)*G2r*(p[9]*psi2self[idx1]*psi2self[idx3]*cx[9]*cy[9]+p[10]*psi2self[idx2]*psi2self[idx4]*cx[10]*cy[10]+p[11]*psi2self[idx1]*psi2self[idx3]*cx[11]*cy[11]+p[12]*psi2self[idx4]*psi2self[idx2]*cx[12]*cy[12]+p[21]*psi2self[idx5]*psi2self[idx7]*cx[21]*cy[21]+p[22]*psi2self[idx6]*psi2self[idx8]*cx[22]*cy[22]+p[23]*psi2self[idx7]*psi2self[idx5]*cx[23]*cy[23]+p[24]*psi2self[idx8]*psi2self[idx6]*cx[24]*cy[24])

      +v(0.25)*G2r*psi2self[idx0]*(p[13]*psi2self[idx13]*cx[13]*cy[13]+p[14]*psi2self[idx14]*cx[14]*cy[14]+p[15]*psi2self[idx15]*cx[15]*cy[15]+p[16]*psi2self[idx16]*cx[16]*cy[16]+p[17]*psi2self[idx17]*cx[17]*cy[17]+p[18]*psi2self[idx18]*cx[18]*cy[18]+p[19]*psi2self[idx19]*cx[19]*cy[19]+p[20]*psi2self[idx20]*cx[20]*cy[20])

      +v(0.25)*G2r*(p[13]*(psi2self[idx5]*psi2self[idx3]+psi2self[idx1]*psi2self[idx7])*cx[13]*cy[13]+p[14]*(psi2self[idx5]*psi2self[idx4]+psi2self[idx2]*psi2self[idx7])*cx[14]*cy[14]+p[15]*(psi2self[idx2]*psi2self[idx8]+psi2self[idx6]*psi2self[idx4])*cx[15]*cy[15]+p[16]*(psi2self[idx6]*psi2self[idx1]+psi2self[idx8]*psi2self[idx3])*cx[16]*cy[16]);



      pyy[idx0]=+v(0.5)*invRho2*g12*rho1pre[idx0]*(ww[1]*rho2pre[idx1]*cy[1]*cy[1]+ww[2]*rho2pre[idx2]*cy[2]*cy[2]+ww[3]*rho2pre[idx3]*cy[3]*cy[3]+ww[4]*rho2pre[idx4]*cy[4]*cy[4]+ww[5]*rho2pre[idx5]*cy[5]*cy[5]+ww[6]*rho2pre[idx6]*cy[6]*cy[6]+ww[7]*rho2pre[idx7]*cy[7]*cy[7]+ww[8]*rho2pre[idx8]*cy[8]*cy[8])

      +v(0.5)*invRho2*g12*rho2pre[idx0]*(ww[1]*rho1pre[idx1]*cy[1]*cy[1]+ww[2]*rho1pre[idx2]*cy[2]*cy[2]+ww[3]*rho1pre[idx3]*cy[3]*cy[3]+ww[4]*rho1pre[idx4]*cy[4]*cy[4]+ww[5]*rho1pre[idx5]*cy[5]*cy[5]+ww[6]*rho1pre[idx6]*cy[6]*cy[6]+ww[7]*rho1pre[idx7]*cy[7]*cy[7]+ww[8]*rho1pre[idx8]*cy[8]*cy[8])

      +v(0.5)*G1a*psi1self[idx0]*(ww[1]*psi1self[idx1]*cy[1]*cy[1]+ww[2]*psi1self[idx2]*cy[2]*cy[2]+ww[3]*psi1self[idx3]*cy[3]*cy[3]+ww[4]*psi1self[idx4]*cy[4]*cy[4]+ww[5]*psi1self[idx5]*cy[5]*cy[5]+ww[6]*psi1self[idx6]*cy[6]*cy[6]+ww[7]*psi1self[idx7]*cy[7]*cy[7]+ww[8]*psi1self[idx8]*cy[8]*cy[8])

      +v(0.5)*G2a*psi2self[idx0]*(ww[1]*psi2self[idx1]*cy[1]*cy[1]+ww[2]*psi2self[idx2]*cy[2]*cy[2]+ww[3]*psi2self[idx3]*cy[3]*cy[3]+ww[4]*psi2self[idx4]*cy[4]*cy[4]+ww[5]*psi2self[idx5]*cy[5]*cy[5]+ww[6]*psi2self[idx6]*cy[6]*cy[6]+ww[7]*psi2self[idx7]*cy[7]*cy[7]+ww[8]*psi2self[idx8]*cy[8]*cy[8])

      +v(0.5)*G1r*psi1self[idx0]*(p[1]*psi1self[idx1]*cy[1]*cy[1]+p[2]*psi1self[idx2]*cy[2]*cy[2]+p[3]*psi1self[idx3]*cy[3]*cy[3]+p[4]*psi1self[idx4]*cy[4]*cy[4]+p[5]*psi1self[idx5]*cy[5]*cy[5]+p[6]*psi1self[idx6]*cy[6]*cy[6]+p[7]*psi1self[idx7]*cy[7]*cy[7]+p[8]*psi1self[idx8]*cy[8]*cy[8])

      +v(0.25)*G1r*psi1self[idx0]*(p[9]*psi1self[idx9]*cy[9]*cy[9]+p[10]*psi1self[idx10]*cy[10]*cy[10]+p[11]*psi1self[idx11]*cy[11]*cy[11]+p[12]*psi1self[idx12]*cy[12]*cy[12]+p[21]*psi1self[idx21]*cy[21]*cy[21]+p[22]*psi1self[idx22]*cy[22]*cy[22]+p[23]*psi1self[idx23]*cy[23]*cy[23]+p[24]*psi1self[idx24]*cy[24]*cy[24])

      +v(0.25)*G1r*(p[9]*psi1self[idx1]*psi1self[idx3]*cy[9]*cy[9]+p[10]*psi1self[idx2]*psi1self[idx4]*cy[10]*cy[10]+p[11]*psi1self[idx1]*psi1self[idx3]*cy[11]*cy[11]+p[12]*psi1self[idx4]*psi1self[idx2]*cy[12]*cy[12]+p[21]*psi1self[idx5]*psi1self[idx7]*cy[21]*cy[21]+p[22]*psi1self[idx6]*psi1self[idx8]*cy[22]*cy[22]+p[23]*psi1self[idx7]*psi1self[idx5]*cy[23]*cy[23]+p[24]*psi1self[idx8]*psi1self[idx6]*cy[24]*cy[24])

      +v(0.25)*G1r*psi1self[idx0]*(p[13]*psi1self[idx13]*cy[13]*cy[13]+p[14]*psi1self[idx14]*cy[14]*cy[14]+p[15]*psi1self[idx15]*cy[15]*cy[15]+p[16]*psi1self[idx16]*cy[16]*cy[16]+p[17]*psi1self[idx17]*cy[17]*cy[17]+p[18]*psi1self[idx18]*cy[18]*cy[18]+p[19]*psi1self[idx19]*cy[19]*cy[19]+p[20]*psi1self[idx20]*cy[20]*cy[20])

      +v(0.25)*G1r*(p[13]*(psi1self[idx5]*psi1self[idx3]+psi1self[idx1]*psi1self[idx7])*cy[13]*cy[13]+p[14]*(psi1self[idx5]*psi1self[idx4]+psi1self[idx2]*psi1self[idx7])*cy[14]*cy[14]+p[15]*(psi1self[idx2]*psi1self[idx8]+psi1self[idx6]*psi1self[idx4])*cy[15]*cy[15]+p[16]*(psi1self[idx6]*psi1self[idx1]+psi1self[idx8]*psi1self[idx3])*cy[16]*cy[16])


      +v(0.5)*G2r*psi2self[idx0]*(p[1]*psi2self[idx1]*cy[1]*cy[1]+p[2]*psi2self[idx2]*cy[2]*cy[2]+p[3]*psi2self[idx3]*cy[3]*cy[3]+p[4]*psi2self[idx4]*cy[4]*cy[4]+p[5]*psi2self[idx5]*cy[5]*cy[5]+p[6]*psi2self[idx6]*cy[6]*cy[6]+p[7]*psi2self[idx7]*cy[7]*cy[7]+p[8]*psi2self[idx8]*cy[8]*cy[8])

      +v(0.25)*G2r*psi2self[idx0]*(p[9]*psi2self[idx9]*cy[9]*cy[9]+p[10]*psi2self[idx10]*cy[10]*cy[10]+p[11]*psi2self[idx11]*cy[11]*cy[11]+p[12]*psi2self[idx12]*cy[12]*cy[12]+p[21]*psi2self[idx21]*cy[21]*cy[21]+p[22]*psi2self[idx22]*cy[22]*cy[22]+p[23]*psi2self[idx23]*cy[23]*cy[23]+p[24]*psi2self[idx24]*cy[24]*cy[24])

      +v(0.25)*G2r*(p[9]*psi2self[idx1]*psi2self[idx3]*cy[9]*cy[9]+p[10]*psi2self[idx2]*psi2self[idx4]*cy[10]*cy[10]+p[11]*psi2self[idx1]*psi2self[idx3]*cy[11]*cy[11]+p[12]*psi2self[idx4]*psi2self[idx2]*cy[12]*cy[12]+p[21]*psi2self[idx5]*psi2self[idx7]*cy[21]*cy[21]+p[22]*psi2self[idx6]*psi2self[idx8]*cy[22]*cy[22]+p[23]*psi2self[idx7]*psi2self[idx5]*cy[23]*cy[23]+p[24]*psi2self[idx8]*psi2self[idx6]*cy[24]*cy[24])

      +v(0.25)*G2r*psi2self[idx0]*(p[13]*psi2self[idx13]*cy[13]*cy[13]+p[14]*psi2self[idx14]*cy[14]*cy[14]+p[15]*psi2self[idx15]*cy[15]*cy[15]+p[16]*psi2self[idx16]*cy[16]*cy[16]+p[17]*psi2self[idx17]*cy[17]*cy[17]+p[18]*psi2self[idx18]*cy[18]*cy[18]+p[19]*psi2self[idx19]*cy[19]*cy[19]+p[20]*psi2self[idx20]*cy[20]*cy[20])

      +v(0.25)*G2r*(p[13]*(psi2self[idx5]*psi2self[idx3]+psi2self[idx1]*psi2self[idx7])*cy[13]*cy[13]+p[14]*(psi2self[idx5]*psi2self[idx4]+psi2self[idx2]*psi2self[idx7])*cy[14]*cy[14]+p[15]*(psi2self[idx2]*psi2self[idx8]+psi2self[idx6]*psi2self[idx4])*cy[15]*cy[15]+p[16]*(psi2self[idx6]*psi2self[idx1]+psi2self[idx8]*psi2self[idx3])*cy[16]*cy[16]);

    }
  }

}

void SNAPSHOT_DELAUNAY(REAL *rho1pre, REAL *rho2pre,
                       REAL *utot, REAL *vtot, int istep){

  static REAL *h_rho1 = NULL;
  static REAL *h_rho2 = NULL;
  static REAL *h_uTot = NULL;
  static REAL *h_vTot = NULL;

  static char fileNameRho1[MAXFILENAME];
  snprintf(fileNameRho1, sizeof(fileNameRho1), "rho1BinaryTime%09d", istep);
  static char fileNameRho2[MAXFILENAME];
  snprintf(fileNameRho2, sizeof(fileNameRho2), "rho2BinaryTime%09d", istep);
  static char fileNameUTot[MAXFILENAME];
  snprintf(fileNameUTot, sizeof(fileNameUTot), "uTotBinaryTime%09d", istep);
  static char fileNameVTot[MAXFILENAME];
  snprintf(fileNameVTot, sizeof(fileNameVTot), "vTotBinaryTime%09d", istep);


  if(h_rho1 == NULL){
    h_rho1 = (REAL *)Malloc((nxdp+2)*(ny+2)*sizeof(REAL));
    h_rho2 = (REAL *)Malloc((nxdp+2)*(ny+2)*sizeof(REAL));
    h_uTot = (REAL *)Malloc((nxdp+2)*(ny+2)*sizeof(REAL));
    h_vTot = (REAL *)Malloc((nxdp+2)*(ny+2)*sizeof(REAL));
  }

  MY_CUDA_CHECK( cudaMemcpyAsync( h_rho1, rho1pre,
                                  (nxdp+2)*(ny+2)*sizeof(REAL),
                                  cudaMemcpyDeviceToHost, stream[0] ) );
  MY_CUDA_CHECK( cudaMemcpyAsync( h_rho2, rho2pre,
                                  (nxdp+2)*(ny+2)*sizeof(REAL),
                                  cudaMemcpyDeviceToHost, stream[0] ) );
  MY_CUDA_CHECK( cudaMemcpyAsync( h_uTot, utot,
                                  (nxdp+2)*(ny+2)*sizeof(REAL),
                                  cudaMemcpyDeviceToHost, stream[0] ) );
  MY_CUDA_CHECK( cudaMemcpyAsync( h_vTot, vtot,
                                  (nxdp+2)*(ny+2)*sizeof(REAL),
                                  cudaMemcpyDeviceToHost, stream[0] ) );

  MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));

  FILE *outRho1 = Fopen(fileNameRho1, "wb");
  FILE *outRho2 = Fopen(fileNameRho2, "wb");
  FILE *outUTot = Fopen(fileNameUTot, "wb");
  FILE *outVTot = Fopen(fileNameVTot, "wb");

  fwrite(&nx, sizeof(int), 1, outRho1);
  fwrite(&ny, sizeof(int), 1, outRho1);
  fwrite(&pbcx, sizeof(int), 1, outRho1);
  fwrite(&pbcy, sizeof(int), 1, outRho1);
  fwrite(h_rho1, sizeof(REAL), (nxdp+2)*(ny+2), outRho1);

  fwrite(&nx, sizeof(int), 1, outRho2);
  fwrite(&ny, sizeof(int), 1, outRho2);
  fwrite(&pbcx, sizeof(int), 1, outRho2);
  fwrite(&pbcy, sizeof(int), 1, outRho2);
  fwrite(h_rho2, sizeof(REAL), (nxdp+2)*(ny+2), outRho2);

  fwrite(&nx, sizeof(int), 1, outUTot);
  fwrite(&ny, sizeof(int), 1, outUTot);
  fwrite(&pbcx, sizeof(int), 1, outUTot);
  fwrite(&pbcy, sizeof(int), 1, outUTot);
  fwrite(h_uTot, sizeof(REAL), (nxdp+2)*(ny+2), outUTot);

  fwrite(&nx, sizeof(int), 1, outVTot);
  fwrite(&ny, sizeof(int), 1, outVTot);
  fwrite(&pbcx, sizeof(int), 1, outVTot);
  fwrite(&pbcy, sizeof(int), 1, outVTot);
  fwrite(h_vTot, sizeof(REAL), (nxdp+2)*(ny+2), outVTot);

  fflush(outRho1);
  fflush(outRho2);
  fflush(outUTot);
  fflush(outVTot);

  fclose(outRho1);
  fclose(outRho2);
  fclose(outUTot);
  fclose(outVTot);

  return;
}

#if defined(SNAPSHOTFROMGPU)
void SNAPSHOT(REAL *deviceRho1pre,REAL *deviceRho2pre, REAL*pxy, REAL *pxykin){


  REAL *rho1pre;
  REAL *rho2pre;

  rho1pre = (REAL*) Malloc((nx+2)*(ny+2)*sizeof(REAL));
  rho2pre = (REAL*) Malloc((nx+2)*(ny+2)*sizeof(REAL));

  Device_SafeMemoryCopyFromDevice(rho1pre, deviceRho1pre, REAL, (nx+2)*(ny+2));
  Device_SafeMemoryCopyFromDevice(rho2pre, deviceRho2pre, REAL, (nx+2)*(ny+2));
#else

void SNAPSHOT(REAL *rho1pre,REAL *rho2pre, REAL*pxy, REAL *pxykin){
#endif

  FILE *fout1, *fout2;
  char filename[MAXFILENAME];


  if(dooutvtk==0){

  snprintf(filename,sizeof(filename),"firstdensity.%d.dat",icount[DENSITY]);
  fout1=Fopen(filename,"w");
  snprintf(filename,sizeof(filename),"firstdensity.dat");
  fout2=Fopen(filename,"w");

  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      fprintf(fout1,"%d %d %e \n",i,j,rho1pre[idx1]);
      fprintf(fout2,"%d %d %e \n",i,j,rho1pre[idx1]);
    }
    fprintf(fout1,"\n");
    fprintf(fout2,"\n");
  }
  fclose(fout1);
  fclose(fout2);


  snprintf(filename,sizeof(filename),"seconddensity.%d.dat",icount[DENSITY]);
  fout1=Fopen(filename,"w");
  snprintf(filename,sizeof(filename),"seconddensity.dat");
  fout2=Fopen(filename,"w");

  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      fprintf(fout1,"%d %d %e \n",i,j,rho2pre[idx1]);
      fprintf(fout2,"%d %d %e \n",i,j,rho2pre[idx1]);
    }
    fprintf(fout1,"\n");
    fprintf(fout2,"\n");
  }
  fclose(fout1);
  fclose(fout2);

  icount[DENSITY]++;

  }


  if(dooutvtk==1) {

        if(noutdens>0 && (istep%noutdens)==0){
          if(dooutvtkrho1 == TRUE){
            snprintf(filename,sizeof(filename),"firstdensity.%d.vtk",icount[VTK]);
            fout1 = fopen(filename, "w");

            fprintf(fout1,"# vtk DataFile Version 2.0\n");
            fprintf(fout1,"CAMPO\n");
            fprintf(fout1,"BINARY\n");
            fprintf(fout1,"DATASET STRUCTURED_POINTS\n");
            fprintf(fout1,"DIMENSIONS %d %d %d\n",nx,ny,1);
            fprintf(fout1,"ORIGIN %d %d %d\n",0,0,0);
            fprintf(fout1,"SPACING 1 1 1\n");
            fprintf(fout1,"POINT_DATA %d\n",nx*ny*1);
            fprintf(fout1,"SCALARS rho1 double\n");
            fprintf(fout1,"LOOKUP_TABLE default\n");

            for (j=1;j<=ny;j++){
              for (i=1;i<=nx;i++){
                idx0=j+(ny+2)*i;
                uint64_t *swapEndianAddr = (uint64_t *)(rho1pre + idx0);
                uint64_t swapEndian = htobe64(*swapEndianAddr);
                fwrite(&swapEndian, sizeof(uint64_t), 1, fout1);
              }
            }

            fclose(fout1);
          }

          if(dooutvtkrho2 == TRUE){
            snprintf(filename,sizeof(filename),"firstdensity2.%d.vtk",icount[VTK]);
            fout2 = fopen(filename, "w");

            fprintf(fout2,"# vtk DataFile Version 2.0\n");
            fprintf(fout2,"CAMPO\n");
            fprintf(fout2,"BINARY\n");
            fprintf(fout2,"DATASET STRUCTURED_POINTS\n");
            fprintf(fout2,"DIMENSIONS %d %d %d\n",nx,ny,1);
            fprintf(fout2,"ORIGIN %d %d %d\n",0,0,0);
            fprintf(fout2,"SPACING 1 1 1\n");
            fprintf(fout2,"POINT_DATA %d\n",nx*ny*1);
            fprintf(fout2,"SCALARS rho1 double 1\n");
            fprintf(fout2,"LOOKUP_TABLE default\n");

            for (j=1;j<=ny;j++){
              for (i=1;i<=nx;i++){
                idx0=j+(ny+2)*i;
                uint64_t *swapEndianAddr = (uint64_t *)(rho2pre + idx0);
                uint64_t swapEndian = htobe64(*swapEndianAddr);
                fwrite(&swapEndian, sizeof(uint64_t), 1, fout2);
              }
            }

            fclose(fout2);
          }
        }

        icount[VTK]++;

  }

#if defined(SNAPSHOTFROMGPU)
  Free(rho1pre);
  Free(rho2pre);
#endif
}

#if defined(SNAPSHOTFROMGPU)
void SNAPSHOT_THERMAL(REAL *deviceRho1pre){

  REAL *rho1pre;

  rho1pre = (REAL*) Malloc((nx+2)*(ny+2)*sizeof(REAL));

  Device_SafeMemoryCopyFromDevice(rho1pre, deviceRho1pre, REAL, (nx+2)*(ny+2));
#else

void SNAPSHOT_THERMAL(REAL *rho1pre){
#endif

  FILE *fout1, *fout2;
  char filename[MAXFILENAME];


  if(dooutvtktemperature==0){

  snprintf(filename,sizeof(filename),"temperature.%d.dat",icount[DENSITY]);
  fout1=Fopen(filename,"w");
  snprintf(filename,sizeof(filename),"temperature.dat");
  fout2=Fopen(filename,"w");

  for(i=1;i<=nx;i++){
    for(j=1;j<=ny;j++){
      idx1=j+(ny+2)*i;
      fprintf(fout1,"%d %d %e \n",i,j,rho1pre[idx1]);
      fprintf(fout2,"%d %d %e \n",i,j,rho1pre[idx1]);
    }
    fprintf(fout1,"\n");
    fprintf(fout2,"\n");
  }
  fclose(fout1);
  fclose(fout2);
  
  if(dooutvtk == 1){
    icount[DENSITY]++;
  }
  
  }
  
  

  if(dooutvtktemperature==1) {

        if(nouttemperature>0 && (istep%nouttemperature)==0){
          if(dooutvtktemperature == TRUE){
            snprintf(filename,sizeof(filename),"temperature.%d.vtk",icount[VTK]);
            fout1 = fopen(filename, "w");

            fprintf(fout1,"# vtk DataFile Version 2.0\n");
            fprintf(fout1,"CAMPO\n");
            fprintf(fout1,"BINARY\n");
            fprintf(fout1,"DATASET STRUCTURED_POINTS\n");
            fprintf(fout1,"DIMENSIONS %d %d %d\n",nx,ny,1);
            fprintf(fout1,"ORIGIN %d %d %d\n",0,0,0);
            fprintf(fout1,"SPACING 1 1 1\n");
            fprintf(fout1,"POINT_DATA %d\n",nx*ny*1);
            fprintf(fout1,"SCALARS rho1 double\n");
            fprintf(fout1,"LOOKUP_TABLE default\n");

            for (j=1;j<=ny;j++){
              for (i=1;i<=nx;i++){
                idx0=j+(ny+2)*i;
                uint64_t *swapEndianAddr = (uint64_t *)(rho1pre + idx0);
                uint64_t swapEndian = htobe64(*swapEndianAddr);
                fwrite(&swapEndian, sizeof(uint64_t), 1, fout1);
              }
            }

            fclose(fout1);
          }

        }
 
	if(dooutvtk == 0){
	  icount[VTK]++;
	}

  }

#if defined(SNAPSHOTFROMGPU)
  Free(rho1pre);
#endif
}

Device_QualKernel void collis_1(pop_type *pF,pop_type *pFeq,REAL *tau){
  REAL omeganow;
  unsigned int idx1;
  unsigned int k;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;
    idx1=j+(sNy2)*i;

    omeganow=(v(1.0)/tau[idx1]);
    
    for(k=0;k<npop;k++){
      pF->p[k][idx1]=pF->p[k][idx1]*(v(1.0)-omeganow)+omeganow*pFeq->p[k][idx1];
    }
   Device_ParallelEndRepeat
}

void collis(pop_type *pF,pop_type *pFeq,REAL *tau){

  Device_ExecuteKernel(collis_1)(pF, pFeq, tau);
    Device_Synchronize();

}

Device_QualKernel void forcingconstructWW_1(REAL *rho1,REAL *rho2, REAL *psi1self,REAL *psi2self){
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2Ny2)
    psi1self[index]=sRho0*(v(1.0)-EXP(-rho1[index]/sRho0));
    psi2self[index]=sRho0*(v(1.0)-EXP(-rho2[index]/sRho0));
  Device_ParallelEndRepeat
}

Device_QualKernel void forcingconstructBC(REAL *rho1,REAL *rho2,
                                          REAL *psi1self,REAL *psi2self){

  Device_ParallelRepeatUntilIndexIsLessThan(2*sNy2)
    psi1self[index]=sRho0*(v(1.0)-EXP(-rho1[index]/sRho0));
    psi2self[index]=sRho0*(v(1.0)-EXP(-rho2[index]/sRho0));
  Device_ParallelEndRepeat
    Device_ParallelRepeatStartingFromAndUntilIndexIsLessThan((sNxdp-2)*sNy2,sNxdp*sNy2)
    psi1self[index]=sRho0*(v(1.0)-EXP(-rho1[index]/sRho0));
    psi2self[index]=sRho0*(v(1.0)-EXP(-rho2[index]/sRho0));
  Device_ParallelEndRepeat
}

Device_QualKernel void forcingconstructBULK(REAL *rho1,REAL *rho2,
                                          REAL *psi1self,REAL *psi2self){
  Device_ParallelRepeatStartingFromAndUntilIndexIsLessThan(2*sNy2,(sNxdp-2)*sNy2)
    psi1self[index]=sRho0*(v(1.0)-EXP(-rho1[index]/sRho0));
    psi2self[index]=sRho0*(v(1.0)-EXP(-rho2[index]/sRho0));
  Device_ParallelEndRepeat
}


 Device_QualKernel void forcingconstructWW_2(REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *psi1self,REAL *psi2self,REAL *utot,REAL *vtot,REAL *tau){
  int plusx1,plusy1;
  int minusx1,minusy1;

  int plusx2,plusy2;
  int minusx2,minusy2;

  REAL Fax,Frx,FXx;
  REAL Fay,Fry,FXy;

  int idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,idx9,
  idx10,idx11,idx12,idx13,idx14,idx15,idx16,idx17,idx18,idx19,
  idx20,idx21,idx22,idx23,idx24;

  REAL invRho2 = v(1.)/(sRho0*sRho0);

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;

    plusx1=i+1;
    minusx1=i-1;
    plusy1=j+1;
    minusy1=j-1;

    plusx2=i+2;
    minusx2=i-2;
    plusy2=j+2;
    minusy2=j-2;

    if(plusy2==(sNy2)){plusy2=sNy+1;}

    if(minusy2==-1){minusy2=0;}

    idx0=j+(sNy2)*i;

    idx1=j+(sNy2)*plusx1;
    idx2=plusy1+(sNy2)*i;
    idx3=j+(sNy2)*minusx1;
    idx4=minusy1+(sNy2)*i;

    idx5=plusy1+(sNy2)*plusx1;
    idx6=plusy1+(sNy2)*minusx1;
    idx7=minusy1+(sNy2)*minusx1;
    idx8=minusy1+(sNy2)*plusx1;

    idx9=j+(sNy2)*plusx2;
    idx10=plusy2+(sNy2)*i;
    idx11=j+(sNy2)*minusx2;
    idx12=minusy2+(sNy2)*i;

    idx13=plusy1+(sNy2)*plusx2;
    idx14=plusy2+(sNy2)*plusx1;
    idx15=plusy2+(sNy2)*minusx1;
    idx16=plusy1+(sNy2)*minusx2;

    idx17=minusy1+(sNy2)*minusx2;
    idx18=minusy2+(sNy2)*minusx1;
    idx19=minusy2+(sNy2)*plusx1;
    idx20=minusy1+(sNy2)*plusx2;

    idx21=plusy2+(sNy2)*plusx2;
    idx22=plusy2+(sNy2)*minusx2;
    idx23=minusy2+(sNy2)*minusx2;
    idx24=minusy2+(sNy2)*plusx2;


    Fax=-sG1a*psi1self[idx0]*(sWw1*psi1self[idx1]-sWw3*psi1self[idx3]+sWw5*psi1self[idx5]-sWw6*psi1self[idx6]-sWw7*psi1self[idx7]+sWw8*psi1self[idx8]);

    Frx=-sG1r*psi1self[idx0]*(sP1*psi1self[idx1]-sP3*psi1self[idx3]+sP5*psi1self[idx5]-sP6*psi1self[idx6]-sP7*psi1self[idx7]+sP8*psi1self[idx8])-sG1r*psi1self[idx0]*(v(2.0)*sP9*psi1self[idx9]+v(2.0)*sP13*psi1self[idx13]+sP14*psi1self[idx14]-sP15*psi1self[idx15]v(-2.0)*sP16*psi1self[idx16]v(-2.0)*sP11*psi1self[idx11]v(-2.0)*sP17*psi1self[idx17]-sP18*psi1self[idx18]+sP19*psi1self[idx19]+v(2.0)*sP20*psi1self[idx20]+v(2.0)*sP21*psi1self[idx21]v(-2.0)*sP22*psi1self[idx22]v(-2.0)*sP23*psi1self[idx23]+v(2.0)*sP24*psi1self[idx24]);

    FXx=-sg12*invRho2*rho1[idx0]*(sWw1*rho2[idx1]-sWw3*rho2[idx3]+sWw5*rho2[idx5]-sWw6*rho2[idx6]-sWw7*rho2[idx7]+sWw8*rho2[idx8]);



    Fay=-sG1a*psi1self[idx0]*(sWw2*psi1self[idx2]-sWw4*psi1self[idx4]+sWw5*psi1self[idx5]+sWw6*psi1self[idx6]-sWw7*psi1self[idx7]-sWw8*psi1self[idx8]);

    Fry=-sG1r*psi1self[idx0]*(sP2*psi1self[idx2]-sP4*psi1self[idx4]+sP5*psi1self[idx5]+sP6*psi1self[idx6]-sP7*psi1self[idx7]-sP8*psi1self[idx8])-sG1r*psi1self[idx0]*(sP13*psi1self[idx13]+v(2.0)*sP14*psi1self[idx14]+v(2.0)*sP10*psi1self[idx10]+v(2.0)*sP15*psi1self[idx15]+sP16*psi1self[idx16]-sP17*psi1self[idx17]v(-2.0)*sP18*psi1self[idx18]v(-2.0)*sP12*psi1self[idx12]v(-2.0)*sP19*psi1self[idx19]-sP20*psi1self[idx20]+v(2.0)*sP21*psi1self[idx21]+v(2.0)*sP22*psi1self[idx22]v(-2.0)*sP23*psi1self[idx23]v(-2.0)*sP24*psi1self[idx24]);

    FXy=-sg12*invRho2*rho1[idx0]*(sWw2*rho2[idx2]-sWw4*rho2[idx4]+sWw5*rho2[idx5]+sWw6*rho2[idx6]-sWw7*rho2[idx7]-sWw8*rho2[idx8]);


    frcex1[idx0]=Fax+Frx+FXx;
    frcey1[idx0]=Fay+Fry+FXy + v(0.5)*(rho1[idx0]+rho2[idx0])*salphaG*temperature[idx0];



    Fax=-sG2a*psi2self[idx0]*(sWw1*psi2self[idx1]-sWw3*psi2self[idx3]+sWw5*psi2self[idx5]-sWw6*psi2self[idx6]-sWw7*psi2self[idx7]+sWw8*psi2self[idx8]);

    Frx=-sG2r*psi2self[idx0]*(sP1*psi2self[idx1]-sP3*psi2self[idx3]+sP5*psi2self[idx5]-sP6*psi2self[idx6]-sP7*psi2self[idx7]+sP8*psi2self[idx8])-sG2r*psi2self[idx0]*(v(2.0)*sP9*psi2self[idx9]+v(2.0)*sP13*psi2self[idx13]+sP14*psi2self[idx14]-sP15*psi2self[idx15]v(-2.0)*sP16*psi2self[idx16]v(-2.0)*sP11*psi2self[idx11]v(-2.0)*sP17*psi2self[idx17]-sP18*psi2self[idx18]+sP19*psi2self[idx19]+v(2.0)*sP20*psi2self[idx20]+v(2.0)*sP21*psi2self[idx21]v(-2.0)*sP22*psi2self[idx22]v(-2.0)*sP23*psi2self[idx23]+v(2.0)*sP24*psi2self[idx24]);

    FXx=-sg12*invRho2*rho2[idx0]*(sWw1*rho1[idx1]-sWw3*rho1[idx3]+sWw5*rho1[idx5]-sWw6*rho1[idx6]-sWw7*rho1[idx7]+sWw8*rho1[idx8]);



    Fay=-sG2a*psi2self[idx0]*(sWw2*psi2self[idx2]-sWw4*psi2self[idx4]+sWw5*psi2self[idx5]+sWw6*psi2self[idx6]-sWw7*psi2self[idx7]-sWw8*psi2self[idx8]);

    Fry=-sG2r*psi2self[idx0]*(sP2*psi2self[idx2]-sP4*psi2self[idx4]+sP5*psi2self[idx5]+sP6*psi2self[idx6]-sP7*psi2self[idx7]-sP8*psi2self[idx8])-sG2r*psi2self[idx0]*(sP13*psi2self[idx13]+v(2.0)*sP14*psi2self[idx14]+v(2.0)*sP10*psi2self[idx10]+v(2.0)*sP15*psi2self[idx15]+sP16*psi2self[idx16]-sP17*psi2self[idx17]v(-2.0)*sP18*psi2self[idx18]v(-2.0)*sP12*psi2self[idx12]v(-2.0)*sP19*psi2self[idx19]-sP20*psi2self[idx20]+v(2.0)*sP21*psi2self[idx21]+v(2.0)*sP22*psi2self[idx22]v(-2.0)*sP23*psi2self[idx23]v(-2.0)*sP24*psi2self[idx24]);

    FXy=-sg12*invRho2*rho2[idx0]*(sWw2*rho1[idx2]-sWw4*rho1[idx4]+sWw5*rho1[idx5]+sWw6*rho1[idx6]-sWw7*rho1[idx7]-sWw8*rho1[idx8]);


    frcex2[idx0]=Fax+Frx+FXx;
    frcey2[idx0]=Fay+Fry+FXy+ v(0.5)*(rho1[idx0]+rho2[idx0])*salphaG*temperature[idx0];

  Device_ParallelEndRepeat

}

 void forcingconstructWW(REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *psi1self,REAL *psi2self,REAL *utot, REAL *vtot, REAL *tau){

  Device_ExecuteKernelAS(forcingconstructBC,0)(rho1+(ny+2), rho2+(ny+2),
                                           psi1self+(ny+2), psi2self+(ny+2));
  Device_ExecuteKernelAS(forcingconstructBULK,1)(rho1+(ny+2), rho2+(ny+2),
                                           psi1self+(ny+2), psi2self+(ny+2));

  ExchangePsiRho(psi1self-(ny+2),psi2self-(ny+2),rho1, rho2, ny+2);

  Device_ExecuteKernel(forcingconstructWW_2)(rho1, rho2, temperature,frcex1, frcey1, frcex2, frcey2, psi1self, psi2self, utot, vtot, tau);
    Device_Synchronize();
}

Device_QualKernel void forcingconstructPBC_1(REAL *rho1,REAL *rho2, REAL *psi1self,REAL *psi2self){
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdp2Ny2)
    psi1self[index]=sRho0*(v(1.0)-EXP(-rho1[index]/sRho0));
    psi2self[index]=sRho0*(v(1.0)-EXP(-rho2[index]/sRho0));
  Device_ParallelEndRepeat
}

 Device_QualKernel void forcingconstructPBC_2(REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *psi1self,REAL *psi2self, REAL *tau){
  int plusx1,plusy1;
  int minusx1,minusy1;

  int plusx2,plusy2;
  int minusx2,minusy2;

  REAL Fax,Frx,FXx;
  REAL Fay,Fry,FXy;

  int idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,idx9,
      idx10,idx11,idx12,idx13,idx14,idx15,idx16,idx17,idx18,idx19,
      idx20,idx21,idx22,idx23,idx24;

  REAL invRho2 = v(1.)/(sRho0*sRho0);

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
  set_i_j; i++; j++;

    plusx1=i+1;
    minusx1=i-1;
    plusy1=j+1;
    minusy1=j-1;

    plusx2=i+2;
    minusx2=i-2;
    plusy2=j+2;
    minusy2=j-2;

    if(plusy1==(sNy1)){plusy1=1;}

    if(minusy1==0){minusy1=sNy;}

    if(plusy2==(sNy1)){plusy2=1;}

    if(minusy2==0){minusy2=sNy;}

    if(plusy2==(sNy2)){plusy2=2;}

    if(minusy2==-1){minusy2=sNy-1;}

    idx0=j+(sNy2)*i;

    idx1=j+(sNy2)*plusx1;
    idx2=plusy1+(sNy2)*i;
    idx3=j+(sNy2)*minusx1;
    idx4=minusy1+(sNy2)*i;

    idx5=plusy1+(sNy2)*plusx1;
    idx6=plusy1+(sNy2)*minusx1;
    idx7=minusy1+(sNy2)*minusx1;
    idx8=minusy1+(sNy2)*plusx1;

    idx9=j+(sNy2)*plusx2;
    idx10=plusy2+(sNy2)*i;
    idx11=j+(sNy2)*minusx2;
    idx12=minusy2+(sNy2)*i;

    idx13=plusy1+(sNy2)*plusx2;
    idx14=plusy2+(sNy2)*plusx1;
    idx15=plusy2+(sNy2)*minusx1;
    idx16=plusy1+(sNy2)*minusx2;

    idx17=minusy1+(sNy2)*minusx2;
    idx18=minusy2+(sNy2)*minusx1;
    idx19=minusy2+(sNy2)*plusx1;
    idx20=minusy1+(sNy2)*plusx2;

    idx21=plusy2+(sNy2)*plusx2;
    idx22=plusy2+(sNy2)*minusx2;
    idx23=minusy2+(sNy2)*minusx2;
    idx24=minusy2+(sNy2)*plusx2;


    Fax=-sG1a*psi1self[idx0]*(sWw1*psi1self[idx1]-sWw3*psi1self[idx3]+sWw5*psi1self[idx5]-sWw6*psi1self[idx6]-sWw7*psi1self[idx7]+sWw8*psi1self[idx8]);

    Frx=-sG1r*psi1self[idx0]*(sP1*psi1self[idx1]-sP3*psi1self[idx3]+sP5*psi1self[idx5]-sP6*psi1self[idx6]-sP7*psi1self[idx7]+sP8*psi1self[idx8])-sG1r*psi1self[idx0]*(v(2.0)*sP9*psi1self[idx9]+v(2.0)*sP13*psi1self[idx13]+sP14*psi1self[idx14]-sP15*psi1self[idx15]v(-2.0)*sP16*psi1self[idx16]v(-2.0)*sP11*psi1self[idx11]v(-2.0)*sP17*psi1self[idx17]-sP18*psi1self[idx18]+sP19*psi1self[idx19]+v(2.0)*sP20*psi1self[idx20]+v(2.0)*sP21*psi1self[idx21]v(-2.0)*sP22*psi1self[idx22]v(-2.0)*sP23*psi1self[idx23]+v(2.0)*sP24*psi1self[idx24]);

    FXx=-sg12*invRho2*rho1[idx0]*(sWw1*rho2[idx1]-sWw3*rho2[idx3]+sWw5*rho2[idx5]-sWw6*rho2[idx6]-sWw7*rho2[idx7]+sWw8*rho2[idx8]);



    Fay=-sG1a*psi1self[idx0]*(sWw2*psi1self[idx2]-sWw4*psi1self[idx4]+sWw5*psi1self[idx5]+sWw6*psi1self[idx6]-sWw7*psi1self[idx7]-sWw8*psi1self[idx8]);

    Fry=-sG1r*psi1self[idx0]*(sP2*psi1self[idx2]-sP4*psi1self[idx4]+sP5*psi1self[idx5]+sP6*psi1self[idx6]-sP7*psi1self[idx7]-sP8*psi1self[idx8])-sG1r*psi1self[idx0]*(sP13*psi1self[idx13]+v(2.0)*sP14*psi1self[idx14]+v(2.0)*sP10*psi1self[idx10]+v(2.0)*sP15*psi1self[idx15]+sP16*psi1self[idx16]-sP17*psi1self[idx17]v(-2.0)*sP18*psi1self[idx18]v(-2.0)*sP12*psi1self[idx12]v(-2.0)*sP19*psi1self[idx19]-sP20*psi1self[idx20]+v(2.0)*sP21*psi1self[idx21]+v(2.0)*sP22*psi1self[idx22]v(-2.0)*sP23*psi1self[idx23]v(-2.0)*sP24*psi1self[idx24]);

    FXy=-sg12*invRho2*rho1[idx0]*(sWw2*rho2[idx2]-sWw4*rho2[idx4]+sWw5*rho2[idx5]+sWw6*rho2[idx6]-sWw7*rho2[idx7]-sWw8*rho2[idx8]);

    frcex1[idx0]=Fax+Frx+FXx;
    frcey1[idx0]=Fay+Fry+FXy + v(0.5)*(rho1[idx0]+rho2[idx0])*salphaG*temperature[idx0];


    Fax=-sG2a*psi2self[idx0]*(sWw1*psi2self[idx1]-sWw3*psi2self[idx3]+sWw5*psi2self[idx5]-sWw6*psi2self[idx6]-sWw7*psi2self[idx7]+sWw8*psi2self[idx8]);

    Frx=-sG2r*psi2self[idx0]*(sP1*psi2self[idx1]-sP3*psi2self[idx3]+sP5*psi2self[idx5]-sP6*psi2self[idx6]-sP7*psi2self[idx7]+sP8*psi2self[idx8])-sG2r*psi2self[idx0]*(v(2.0)*sP9*psi2self[idx9]+v(2.0)*sP13*psi2self[idx13]+sP14*psi2self[idx14]-sP15*psi2self[idx15]v(-2.0)*sP16*psi2self[idx16]v(-2.0)*sP11*psi2self[idx11]v(-2.0)*sP17*psi2self[idx17]-sP18*psi2self[idx18]+sP19*psi2self[idx19]+v(2.0)*sP20*psi2self[idx20]+v(2.0)*sP21*psi2self[idx21]v(-2.0)*sP22*psi2self[idx22]v(-2.0)*sP23*psi2self[idx23]+v(2.0)*sP24*psi2self[idx24]);

    FXx=-sg12*invRho2*rho2[idx0]*(sWw1*rho1[idx1]-sWw3*rho1[idx3]+sWw5*rho1[idx5]-sWw6*rho1[idx6]-sWw7*rho1[idx7]+sWw8*rho1[idx8]);



    Fay=-sG2a*psi2self[idx0]*(sWw2*psi2self[idx2]-sWw4*psi2self[idx4]+sWw5*psi2self[idx5]+sWw6*psi2self[idx6]-sWw7*psi2self[idx7]-sWw8*psi2self[idx8]);

    Fry=-sG2r*psi2self[idx0]*(sP2*psi2self[idx2]-sP4*psi2self[idx4]+sP5*psi2self[idx5]+sP6*psi2self[idx6]-sP7*psi2self[idx7]-sP8*psi2self[idx8])-sG2r*psi2self[idx0]*(sP13*psi2self[idx13]+v(2.0)*sP14*psi2self[idx14]+v(2.0)*sP10*psi2self[idx10]+v(2.0)*sP15*psi2self[idx15]+sP16*psi2self[idx16]-sP17*psi2self[idx17]v(-2.0)*sP18*psi2self[idx18]v(-2.0)*sP12*psi2self[idx12]v(-2.0)*sP19*psi2self[idx19]-sP20*psi2self[idx20]+v(2.0)*sP21*psi2self[idx21]+v(2.0)*sP22*psi2self[idx22]v(-2.0)*sP23*psi2self[idx23]v(-2.0)*sP24*psi2self[idx24]);

    FXy=-sg12*invRho2*rho2[idx0]*(sWw2*rho1[idx2]-sWw4*rho1[idx4]+sWw5*rho1[idx5]+sWw6*rho1[idx6]-sWw7*rho1[idx7]-sWw8*rho1[idx8]);


    frcex2[idx0]=Fax+Frx+FXx;
    frcey2[idx0]=Fay+Fry+FXy+ v(0.5)*(rho1[idx0]+rho2[idx0])*salphaG*temperature[idx0];

  Device_ParallelEndRepeat
}

 void forcingconstructPBC(REAL *rho1,REAL *rho2,REAL *temperature,REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *psi1self,REAL *psi2self, REAL *tau){


  Device_ExecuteKernelAS(forcingconstructBC,0)(rho1+(ny+2), rho2+(ny+2),
                                           psi1self+(ny+2), psi2self+(ny+2));
  Device_ExecuteKernelAS(forcingconstructBULK,1)(rho1+(ny+2), rho2+(ny+2),
                                           psi1self+(ny+2), psi2self+(ny+2));

  ExchangePsiRho(psi1self-(ny+2),psi2self-(ny+2), rho1, rho2, ny+2);

  Device_ExecuteKernel(forcingconstructPBC_2)(rho1, rho2, temperature, frcex1, frcey1,
					      frcex2, frcey2, psi1self, psi2self, tau);
  Device_Synchronize();
}

Device_QualKernel void setTau_1(REAL *tau, REAL Relax){
  unsigned int idx1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;

    idx1=j+(sNy2)*i;

    tau[idx1]=Relax;

  Device_ParallelEndRepeat
}

void setTau(REAL *tau, REAL relax){

  Device_ExecuteKernel(setTau_1)(tau, relax);
    Device_Synchronize();

}
 
 Device_QualKernel void computeTauAndUeq_1(REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *u1pre,REAL *u2pre,REAL *v1pre,REAL *v2pre,REAL *rho1pre,REAL *rho2pre,REAL *eqfieldx1,REAL *eqfieldy1,REAL *eqfieldx2,REAL *eqfieldy2,REAL *tau){
  REAL momx, momy, invrho;
  unsigned int idx1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;

    idx1=j+(sNy2)*i;

    tau[idx1]=sRelax1;

    invrho=v(1.0)/(rho1pre[idx1]+rho2pre[idx1]);
    momx=(rho1pre[idx1]*u1pre[idx1]+rho2pre[idx1]*u2pre[idx1])*invrho;
    momy=(rho1pre[idx1]*v1pre[idx1]+rho2pre[idx1]*v2pre[idx1])*invrho;

    eqfieldx1[idx1]=momx+tau[idx1]*frcex1[idx1]/(rho1pre[idx1]);

    eqfieldy1[idx1]=momy+tau[idx1]*frcey1[idx1]/(rho1pre[idx1]);

    eqfieldx2[idx1]=momx+tau[idx1]*frcex2[idx1]/(rho2pre[idx1]);

    eqfieldy2[idx1]=momy+tau[idx1]*frcey2[idx1]/(rho2pre[idx1]);
  Device_ParallelEndRepeat
}

 void computeTauAndUeq(REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *u1pre,REAL *u2pre,REAL *v1pre,REAL *v2pre,REAL *rho1pre,REAL *rho2pre,REAL *eqfieldx1,REAL *eqfieldy1,REAL *eqfieldx2,REAL *eqfieldy2,REAL *tau){

  Device_ExecuteKernel(computeTauAndUeq_1)(frcex1, frcey1, frcex2, frcey2, u1pre, u2pre, v1pre, v2pre, rho1pre, rho2pre, eqfieldx1, eqfieldy1, eqfieldx2, eqfieldy2, tau);
    Device_Synchronize();

}

Device_QualKernel void computeTauAndUeq_1_thermal(REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *u1pre,REAL *u2pre,REAL *v1pre,REAL *v2pre,REAL *rho1pre,REAL *rho2pre,REAL *eqfieldx1,REAL *eqfieldy1){
  REAL  momx, momy,invrho;
  unsigned int idx1;

  declare_i_j;
  Device_ParallelRepeatUntilIndexIsLessThan(sNxdpNy)
    set_i_j; i++; j++;

    idx1=j+(sNy2)*i;

    invrho=v(1.0)/(rho1pre[idx1]+rho2pre[idx1]);

    momx=(rho1pre[idx1]*u1pre[idx1]+rho2pre[idx1]*u2pre[idx1])*invrho;
    momy=(rho1pre[idx1]*v1pre[idx1]+rho2pre[idx1]*v2pre[idx1])*invrho;

    eqfieldx1[idx1]=momx+v(0.5)*invrho*(frcex1[idx1]+frcex2[idx1]);
    eqfieldy1[idx1]=momy+v(0.5)*invrho*(frcey1[idx1]+frcey2[idx1]);

  Device_ParallelEndRepeat
}

 void computeTauAndUeq_thermal(REAL *frcex1,REAL *frcey1,REAL *frcex2,REAL *frcey2,REAL *u1pre,REAL *u2pre,REAL *v1pre,REAL *v2pre,REAL *rho1pre,REAL *rho2pre,REAL *eqfieldx1,REAL *eqfieldy1){

   Device_ExecuteKernel(computeTauAndUeq_1_thermal)(frcex1, frcey1, frcex2, frcey2, u1pre, u2pre, v1pre, v2pre, rho1pre, rho2pre, eqfieldx1, eqfieldy1);
    Device_Synchronize();

}

void dellar(){

  REAL kappa;

  DeviceEnableLoadInt32ValuesForBlock();
  DeviceEnableLoadREALValuesForBlock();

  Device_SafeLoadInt32ValueOnDevice(sNx,            nx);
  Device_SafeLoadInt32ValueOnDevice(sNxdp,          nxdp);
  Device_SafeLoadInt32ValueOnDevice(sNy,            ny);
  Device_SafeLoadInt32ValueOnDevice(sNxdp2Ny2,      (nxdp+2)*(ny+2));
  Device_SafeLoadInt32ValueOnDevice(sNxdpNy,        (nxdp)*(ny));
  Device_SafeLoadInt32ValueOnDevice(sNx2  ,         (nx+2));
  Device_SafeLoadInt32ValueOnDevice(sNxdp2,         (nxdp+2));
  Device_SafeLoadInt32ValueOnDevice(sNy2,           (ny+2));
  Device_SafeLoadInt32ValueOnDevice(sNxdp1,         (nxdp+1));
  Device_SafeLoadInt32ValueOnDevice(sNx1,           (nx+1));
  Device_SafeLoadInt32ValueOnDevice(sNy1,           (ny+1));

  Device_SafeLoadREALValueOnDevice(sG1a,         G1a);
  Device_SafeLoadREALValueOnDevice(sG1r,         G1r);
  Device_SafeLoadREALValueOnDevice(sG2a,         G2a);
  Device_SafeLoadREALValueOnDevice(sG2r,         G2r);
  Device_SafeLoadREALValueOnDevice(sG12,         G12);
  Device_SafeLoadREALValueOnDevice(salphaG,      alphaG);
  Device_SafeLoadREALValueOnDevice(sPERTURB_VELO_AMPLITUDE, PERTURB_VELO_AMPLITUDE);

  Device_SafeLoadREALValueOnDevice(sRt0,         v(4.0)/v(9.0));
  Device_SafeLoadREALValueOnDevice(sRt1,         v(1.0)/v(9.0));
  Device_SafeLoadREALValueOnDevice(sRt2,         v(1.0)/v(36.0));

  Device_SafeLoadREALValueOnDevice(sRhowall1,       rhowall1);
  Device_SafeLoadREALValueOnDevice(sRhowall2,       rhowall2);
  Device_SafeLoadREALValueOnDevice(sRho0,        rho0);

  Device_SafeLoadREALValueOnDevice(sUwallup,       uwallup);
  Device_SafeLoadREALValueOnDevice(sUwalldown,     uwalldown);

  Device_SafeLoadREALValueOnDevice(sRelax1,       relax1);
  Device_SafeLoadREALValueOnDevice(sRelax2,       relax2);
  Device_SafeLoadREALValueOnDevice(sRelaxG,       relaxG);

  kappa=2*ACOS(-1.0)/(ny-1);

  Device_SafeLoadREALValueOnDevice(sKappa,       kappa);

  Device_SafeLoadREALValueOnDevice(sWw0,         v(16.0)/v(36.0));
  Device_SafeLoadREALValueOnDevice(sWw1,         v(4.0)/v(36.0));
  Device_SafeLoadREALValueOnDevice(sWw2,         v(4.0)/v(36.0));
  Device_SafeLoadREALValueOnDevice(sWw3,         v(4.0)/v(36.0));
  Device_SafeLoadREALValueOnDevice(sWw4,         v(4.0)/v(36.0));
  Device_SafeLoadREALValueOnDevice(sWw5,         v(1.0)/v(36.0));
  Device_SafeLoadREALValueOnDevice(sWw6,         v(1.0)/v(36.0));
  Device_SafeLoadREALValueOnDevice(sWw7,         v(1.0)/v(36.0));
  Device_SafeLoadREALValueOnDevice(sWw8,         v(1.0)/v(36.0));

  Device_SafeLoadREALValueOnDevice(sP0,          v(247.0)/v(420.0));
  Device_SafeLoadREALValueOnDevice(sP1,          v(4.0)/v(63.0));
  Device_SafeLoadREALValueOnDevice(sP2,          v(4.0)/v(63.0));
  Device_SafeLoadREALValueOnDevice(sP3,          v(4.0)/v(63.0));
  Device_SafeLoadREALValueOnDevice(sP4,          v(4.0)/v(63.0));
  Device_SafeLoadREALValueOnDevice(sP5,          v(4.0)/v(135.0));
  Device_SafeLoadREALValueOnDevice(sP6,          v(4.0)/v(135.0));
  Device_SafeLoadREALValueOnDevice(sP7,          v(4.0)/v(135.0));
  Device_SafeLoadREALValueOnDevice(sP8,          v(4.0)/v(135.0));
  Device_SafeLoadREALValueOnDevice(sP9,          v(1.0)/v(180.0));
  Device_SafeLoadREALValueOnDevice(sP10,         v(1.0)/v(180.0));
  Device_SafeLoadREALValueOnDevice(sP11,         v(1.0)/v(180.0));
  Device_SafeLoadREALValueOnDevice(sP12,         v(1.0)/v(180.0));
  Device_SafeLoadREALValueOnDevice(sP13,         v(2.0)/v(945.0));
  Device_SafeLoadREALValueOnDevice(sP14,         v(2.0)/v(945.0));
  Device_SafeLoadREALValueOnDevice(sP15,         v(2.0)/v(945.0));
  Device_SafeLoadREALValueOnDevice(sP16,         v(2.0)/v(945.0));
  Device_SafeLoadREALValueOnDevice(sP17,         v(2.0)/v(945.0));
  Device_SafeLoadREALValueOnDevice(sP18,         v(2.0)/v(945.0));
  Device_SafeLoadREALValueOnDevice(sP19,         v(2.0)/v(945.0));
  Device_SafeLoadREALValueOnDevice(sP20,         v(2.0)/v(945.0));
  Device_SafeLoadREALValueOnDevice(sP21,         v(1.0)/v(15120.0));
  Device_SafeLoadREALValueOnDevice(sP22,         v(1.0)/v(15120.0));
  Device_SafeLoadREALValueOnDevice(sP23,         v(1.0)/v(15120.0));
  Device_SafeLoadREALValueOnDevice(sP24,         v(1.0)/v(15120.0));


  Device_SafeLoadREALValueOnDevice(sCx0,         v(0.0));
  Device_SafeLoadREALValueOnDevice(sCy0,         v(0.0));

  Device_SafeLoadREALValueOnDevice(sCx1,         v(1.0));
  Device_SafeLoadREALValueOnDevice(sCy1,         v(0.0));

  Device_SafeLoadREALValueOnDevice(sCx2,         v(0.0));
  Device_SafeLoadREALValueOnDevice(sCy2,         v(1.0));

  Device_SafeLoadREALValueOnDevice(sCx3,        v(-1.0));
  Device_SafeLoadREALValueOnDevice(sCy3,        v(-0.0));

  Device_SafeLoadREALValueOnDevice(sCx4,        v(-0.0));
  Device_SafeLoadREALValueOnDevice(sCy4,        v(-1.0));

  Device_SafeLoadREALValueOnDevice(sCx5,         v(1.0));
  Device_SafeLoadREALValueOnDevice(sCy5,         v(1.0));

  Device_SafeLoadREALValueOnDevice(sCx6,        v(-1.0));
  Device_SafeLoadREALValueOnDevice(sCy6,         v(1.0));

  Device_SafeLoadREALValueOnDevice(sCx7,        v(-1.0));
  Device_SafeLoadREALValueOnDevice(sCy7,        v(-1.0));

  Device_SafeLoadREALValueOnDevice(sCx8,         v(1.0));
  Device_SafeLoadREALValueOnDevice(sCy8,        v(-1.0));


  Device_SafeLoadREALValueOnDevice(sCx9,         v(2.0));
  Device_SafeLoadREALValueOnDevice(sCy9,         v(0.0));

  Device_SafeLoadREALValueOnDevice(sCx10,        v(0.0));
  Device_SafeLoadREALValueOnDevice(sCy10,        v(2.0));

  Device_SafeLoadREALValueOnDevice(sCx11,       v(-2.0));
  Device_SafeLoadREALValueOnDevice(sCy11,        v(0.0));

  Device_SafeLoadREALValueOnDevice(sCx12,        v(0.0));
  Device_SafeLoadREALValueOnDevice(sCy12,       v(-2.0));

  Device_SafeLoadREALValueOnDevice(sCx13,        v(2.0));
  Device_SafeLoadREALValueOnDevice(sCy13,        v(1.0));

  Device_SafeLoadREALValueOnDevice(sCx14,        v(1.0));
  Device_SafeLoadREALValueOnDevice(sCy14,        v(2.0));

  Device_SafeLoadREALValueOnDevice(sCx15,       v(-1.0));
  Device_SafeLoadREALValueOnDevice(sCy15,        v(2.0));

  Device_SafeLoadREALValueOnDevice(sCx16,       v(-2.0));
  Device_SafeLoadREALValueOnDevice(sCy16,        v(1.0));

  Device_SafeLoadREALValueOnDevice(sCx17,       v(-2.0));
  Device_SafeLoadREALValueOnDevice(sCy17,       v(-1.0));

  Device_SafeLoadREALValueOnDevice(sCx18,       v(-1.0));
  Device_SafeLoadREALValueOnDevice(sCy18,       v(-2.0));

  Device_SafeLoadREALValueOnDevice(sCx19,        v(1.0));
  Device_SafeLoadREALValueOnDevice(sCy19,       v(-2.0));

  Device_SafeLoadREALValueOnDevice(sCx20,        v(2.0));
  Device_SafeLoadREALValueOnDevice(sCy20,       v(-1.0));

  Device_SafeLoadREALValueOnDevice(sCx21,        v(2.0));
  Device_SafeLoadREALValueOnDevice(sCy21,        v(2.0));

  Device_SafeLoadREALValueOnDevice(sCx22,       v(-2.0));
  Device_SafeLoadREALValueOnDevice(sCy22,        v(2.0));

  Device_SafeLoadREALValueOnDevice(sCx23,       v(-2.0));
  Device_SafeLoadREALValueOnDevice(sCy23,       v(-2.0));

  Device_SafeLoadREALValueOnDevice(sCx24,        v(2.0));
  Device_SafeLoadREALValueOnDevice(sCy24,       v(-2.0));
#if defined(USEMPI)  
  {
        int tid;
        MPI_Comm_rank(MPI_COMM_WORLD,&tid);
        Device_SafeLoadInt32ValueOnDevice(sTid,            tid);
  }
#endif


  ww[0]=16./36.;
  ww[1]=4./36.;
  ww[2]=4./36.;
  ww[3]=4./36.;
  ww[4]=4./36.;
  ww[5]=1./36.;
  ww[6]=1./36.;
  ww[7]=1./36.;
  ww[8]=1./36.;


  p[0]=247./420.;
  p[1]=4./63.;
  p[2]=4./63.;
  p[3]=4./63.;
  p[4]=4./63.;
  p[5]=4./135.;
  p[6]=4./135.;
  p[7]=4./135.;
  p[8]=4./135.;
  p[9]=1./180.;
  p[10]=1./180.;
  p[11]=1./180.;
  p[12]=1./180.;
  p[13]=2./945.;
  p[14]=2./945.;
  p[15]=2./945.;
  p[16]=2./945.;
  p[17]=2./945.;
  p[18]=2./945.;
  p[19]=2./945.;
  p[20]=2./945.;
  p[21]=1./15120.;
  p[22]=1./15120.;
  p[23]=1./15120.;
  p[24]=1./15120.;


  cx[0]=0.;
  cy[0]=0.;

  cx[1]=1.;
  cy[1]=0.;

  cx[2]=0.;
  cy[2]=1.;

  cx[3]=-1.;
  cy[3]=-0.;

  cx[4]=-0.;
  cy[4]=-1.;

  cx[5]=1.;
  cy[5]=1.;

  cx[6]=-1.;
  cy[6]=1.;

  cx[7]=-1.;
  cy[7]=-1.;

  cx[8]=1.;
  cy[8]=-1.;


  cx[9]=2.;
  cy[9]=0.;

  cx[10]=0.;
  cy[10]=2.;

  cx[11]=-2.;
  cy[11]=0.;

  cx[12]=0.;
  cy[12]=-2.;

  cx[13]=2.;
  cy[13]=1.;

  cx[14]=1.;
  cy[14]=2.;

  cx[15]=-1.;
  cy[15]=2.;

  cx[16]=-2.;
  cy[16]=1.;

  cx[17]=-2.;
  cy[17]=-1.;

  cx[18]=-1.;
  cy[18]=-2.;

  cx[19]=1.;
  cy[19]=-2.;

  cx[20]=2.;
  cy[20]=-1.;

  cx[21]=2.;
  cy[21]=2.;

  cx[22]=-2.;
  cy[22]=2.;

  cx[23]=-2.;
  cy[23]=-2.;

  cx[24]=2.;
  cy[24]=-2.;

}

 void flagging(int *flag, int nxhost){

  FILE *fout;

for(i=0;i<=nxhost+1;i++){
  for(j=0;j<=ny+1;j++){
    idx1=j+(ny+2)*i;
    flag[idx1]=1;
  }
}

for(i=0;i<=nxhost+1;i++){
 idx1=0+(ny+2)*i;
 idx2=ny+1+(ny+2)*i;
 flag[idx1]=0;
 flag[idx2]=0;
}

   fout=Fopen("flag.dat","r");
   int iscratch, jscratch;
   for(i=0;i<=nxhost+1;i++){
     for(j=0;j<=ny+1;j++){
       idx1=j+(ny+2)*i;
       fscanf(fout,"%d %d %d\n",&iscratch,&jscratch,&flag[idx1]);
     }
   }
   fclose(fout);

   countflag=0.;
   for(i=1;i<=nxhost;i++){
     for(j=1;j<=ny;j++){
       idx1=j+(ny+2)*i;
       if(flag[idx1]==1){
         countflag=countflag+1.0;
       }
     }
   }

}




 int main(int argc, char *argv[]) {

  int dTrigger = 0;

  pop_type feq1;

  pop_type feq2;

  pop_type geq;

  pop_type *pF1Source;
  pop_type *pF2Source;
  pop_type *pGSource;
  
  pop_type *pF1Dest;
  pop_type *pF2Dest;
  pop_type *pGDest;

  pop_type memoryup1,memoryup2, memoryupG, boundaryupG;
  pop_type memorydown1,memorydown2, memorydownG, boundarydownG;

  pop_type *deviceStructF1;
  pop_type *deviceStructFeq1;

  pop_type *deviceStructF2;
  pop_type *deviceStructFeq2;

  pop_type *deviceStructG;
  pop_type *deviceStructGeq;
  
  pop_type *deviceStructF1Buf;
  pop_type *deviceStructF2Buf;
  pop_type *deviceStructGBuf;

  pop_type *deviceStructF1Source;
  pop_type *deviceStructF2Source;
  pop_type *deviceStructGSource;

  pop_type *deviceStructF1Dest;
  pop_type *deviceStructF2Dest;
  pop_type *deviceStructGDest;

  pop_type *deviceStructMemoryup1;
  pop_type *deviceStructMemoryup2;
  pop_type *deviceStructMemoryupG;
  pop_type *deviceStructBndUpG;

  pop_type *deviceStructMemorydown1;
  pop_type *deviceStructMemorydown2;
  pop_type *deviceStructMemorydownG;
  pop_type *deviceStructBndDownG;

  pop_type *deviceTemporaneoF1;
  pop_type *deviceTemporaneoF2;

  int  *flag;
  int  *d_flag;
  REAL *u1pre,*v1pre;
  REAL *u1post,*v1post;

  REAL *u2pre,*v2pre;
  REAL *u2post,*v2post;

  REAL *rho1pre,*rho2pre, *temperaturepre;
  REAL *rho1post,*rho2post, *temperaturepost;

  REAL *frcex1,*frcex2;
  REAL *frcey1,*frcey2;

  REAL *eqfieldx1,*eqfieldx2, *eqfieldxG;
  REAL *eqfieldy1,*eqfieldy2, *eqfieldyG;

  REAL *psi1self,*psi2self;

  REAL *utot,*vtot;

  REAL *pxx,*pyy,*pxy;

  REAL *pxy11,*pxy12,*pxy22;

  REAL *pxxkin,*pyykin,*pxykin;

  REAL *tau, *tauG;

  REAL uwalluptd, uwalldowntd; 

  int i, scratch;
  int nsteps, nout, noutconfig, noutconfigMod, noutconfigCount=0;
  int initcond=CONST;
  int initcondG=0;

  char *inputfile=NULL;

  char *po; 
  dictionary *ini;
  char key[MAXSTRLEN];

  unsigned int buffered = 0;

  int myid=0, myrank=0, nxhost;
#if defined(USEMPI)
  int iarr;
  int ifirst=1;
  REAL *tp[NATB];  
#endif  

  int nstreams=2;
  
  TIMER_DEF;
  double tsimtime=0.;

  int useTHERMAL=TRUE;
  
  int postPreparation=TRUE;

  for(i = 1; i < argc; i++) {
    po = argv[i];
    if (*po++ == '-') {
      switch (*po++) {
        case 'h':
          SKIPBLANK
          Usage(argv[0]);
          exit(OK);
          break;
        case 'v':
          SKIPBLANK
          verbose=TRUE;
          break;
        case 'i':
          SKIPBLANK
          inputfile=Strdup(po);
          break;
        default:
          Usage(argv[0]);
          exit(OK);
          break;
      }
    }
  }
#if defined(MPI)
  StartMpi(&myid, &nprocs,&argc,&argv);
  if(nprocs>1) {
          assignDeviceToProcess(&myrank);
  }
  if(nprocs>1) {
    monogpu=FALSE;
  }
#endif  
  if(ngpu==0) {
     writelog(TRUE,APPLICATION_RC,"NGPU must be explicitly set\n");
  }

  if(myrank>0) {  /* This needs to be done for tasks that would use GPUS
                   starting from an index not equal to zero */
    char scratch[MAXSTRLEN];
    snprintf(scratch,sizeof(scratch),"%d",myrank*ngpu);
    for(int i=1; i<ngpu; i++) {
      snprintf(scratch+strlen(scratch),sizeof(scratch)-strlen(scratch),",%d",
               (myrank)*ngpu+i);
    }
    printf("Task %d(%d), setting CUDA_VISIBLE_DEVICES to %s\n",myid,myrank,scratch);
    if(setenv("CUDA_VISIBLE_DEVICES",scratch,1)<0) {
      writelog(TRUE,APPLICATION_RC,"setting CUDA_VISIBLE_DEVICES");
    }
  }


  if(inputfile==NULL) {
    writelog(TRUE,APPLICATION_RC,"no inputfile");
  }

  ini = iniparser_load(inputfile);
  if (ini==NULL) {
    writelog(TRUE,APPLICATION_RC,"cannot parse file: %s\n", inputfile);
  }

  READINTFI(nx,"nx");

  nxdp=nx/nprocs;
  if(nxdp*nprocs!=nx) {
    writelog(TRUE,APPLICATION_RC,"The number of processors %d is not an exact divider of the size %d",nprocs,nx);
  }
  READINTFI(useTHERMAL,"THERMAL");
    
  READINTFI(pbcx,"periodic boundary condition along x");
  innx=1./(REAL)nx;
  innxm2=1./(REAL)(nx-2);

  READINTFI(ny,"ny");
  innxny=1./(REAL)(nx*ny);

  READINTFI(pbcy,"periodic boundary condition along y");
  READINTFI(roughWallDown,"roughWallDown");
  READINTFI(roughWallUp,"roughWallUp");

  READINTFI(nsteps,"nsteps");
#if 0
  READINTFI(nout,"nout");
#endif
  READINTFI(noutdens,"nout density");
  READINTFI(nouttemperature,"nout temperature");
  READINTFI(noutenergy,"nout energy");
  READINTFI(dooutvtk,"write vtk file");
  READINTFI(dooutvtkrho1,"write vtk file rho1");
  READINTFI(dooutvtkrho2,"write vtk file rho2");
  READINTFI(dooutvtktemperature,"write vtk file temperature");
  READINTFI(dooutenergy,"write energy file");

  READINTFI(noutvelo,"nout velocity");
  READINTFI(nouttens,"nout tensor");
  READINTFI(noutave, "nout average");
  READINTFI(noutconfig,"noutconfig");
  READINTFI(noutconfigMod,"noutconfigMod");
  READINTFI(ncheckdelaunay,"ncheckdelaunay");
  READINTFI(nmindelaunay,"nmindelaunay");
  READINTFI(delaunayDebug,"delaunayDebug");
  READREALFI(bubbleThreshold,"bubbleThreshold");

  READINTFI(scratch,"start from scratch");
  READINTFI(postPreparation,"post preparation temp");

  READINTFI(initcond,"droplet initialization");
  READINTFI(initcondG,"temperature initialization");
  READINTFI(NUMx,"number of droplet x");

  READINTFI(NUMy,"number of droplet y");

  READREALFI(WD,"WD");
  READREALFI(threshold_WD,"threshold_WD");

  READREALFI(diameter,"diameter");

  READREALFI(spacing,"spacing");

  READREALFI(uwallup,"uWallUp");
  READREALFI(uwalldown,"uWallDown");

  READREALFI(rhowall1,"rhoWallMax");
  READREALFI(rhowall2,"rhoWallMin");
  READREALFI(relax1,"tau");
  READREALFI(relaxG,"tauG");
  READREALFI(rhol,"rhoMax");
  READREALFI(rhog,"rhoMin");
  READREALFI(rho0,"rho0");
  READREALFI(G1a,"G11a");
  READREALFI(G2a,"G22a");
  READREALFI(G1r,"G11r");
  READREALFI(G2r,"G22r");
  READREALFI(G12,"G12");
  READREALFI(alphaG,"alphaG");
  READREALFI(Tup,"Tup");
  READREALFI(Tdown,"Tdown");
  READREALFI(PERTURB_VELO_AMPLITUDE,"initial velocity perturbation");

  READINTFI(randseed,"random seed");

  CUDAREADINTFI(whichgpu,"Which GPU");
  CUDAREADINTFI(sThreadsNum,"nthreads");
  CUDAREADINTFI(sBlocksNum,"nblocks");

  if(ini) { iniparser_freedict(ini); }

  for(i=0; i<LASTOBS; i++) {
      icount[i]=0;
  }

  if(initcond == HONEYFMMM){
    int H = (int)(ceil((diameter + 2.*spacing + 0.5*sqrt(3.)*((double)(NUMy - 1))*(diameter + spacing))));
    if(ny != H){
      fprintf(stderr, "No right y-size to build the honeycomb initial configuration!\nYou need ny = %d\n", H);
      return 1;
    }
    
    int L = (int)(ceil((diameter*NUMx + spacing*(double)(NUMx + 1))));
    if(nx != L){
      fprintf(stderr, "No right x-size to build the honeycomb initial configuration!\nYou need nx = %d\n", L);
      return 1;
    }
  }

  nout = noutdens;
  if(noutvelo > 0 && noutvelo<nout){
    nout=noutvelo;
  }
  if(nouttens > 0 && nouttens<nout){
    nout=nouttens;
  }
  if(noutave > 0 && noutave<nout){
    nout=noutave;
  }
  if(nouttemperature%nout) {
      writelog(TRUE,APPLICATION_RC,"nout temperature is not an exact multiplier of nout\n!");
  }

  if(noutdens%nout) {
      writelog(TRUE,APPLICATION_RC,"nout density is not an exact multiplier of nout\n!");
  }

  if(noutvelo%nout) {
      writelog(TRUE,APPLICATION_RC,"nout velocity is not an exact multiplier of nout\n!");
  }

  if(nouttens%nout) {
    writelog(TRUE,APPLICATION_RC,"nout tensor (%d) is not an exact multiplier of nout (%d)\n!",nouttens,nout);
  }

  if(noutave%nout) {
    writelog(TRUE,APPLICATION_RC,"nout average (%d) is not an exact multiplier of nout (%d)\n!",noutave,nout);
  }


  if(nprocs>1) {
    if(noutconfig%nout) {
      writelog(TRUE,APPLICATION_RC,"For the MPI version noutconfig must be an exact multiplier of nout!");
    }
  }

  if(nprocs==1) {
          Device_InitDevice(whichgpu);
  }

  {
        int rwg=-1;
        cudaGetDevice(&rwg);
        printf("Task %d: Using gpu number %d\n",myid, rwg);
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryAlloc(f1.p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryAlloc(feq1.p[i], REAL, (nxdp+2)*(ny+2));

    Device_SafeMemoryAlloc(f2.p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryAlloc(feq2.p[i], REAL, (nxdp+2)*(ny+2));

    Device_SafeMemoryAlloc(g.p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryAlloc(geq.p[i], REAL, (nxdp+2)*(ny+2));

    Device_SafeMemoryAlloc(memoryup1.p[i], REAL, (nxdp+2));
    Device_SafeMemoryAlloc(memoryup2.p[i], REAL, (nxdp+2));
    Device_SafeMemoryAlloc(memoryupG.p[i], REAL, (nxdp+2));
    Device_SafeMemoryAlloc(boundaryupG.p[i], REAL, (nxdp+2));

    Device_SafeMemoryAlloc(memorydown1.p[i], REAL, (nxdp+2));
    Device_SafeMemoryAlloc(memorydown2.p[i], REAL, (nxdp+2));
    Device_SafeMemoryAlloc(memorydownG.p[i], REAL, (nxdp+2));
    Device_SafeMemoryAlloc(boundarydownG.p[i], REAL, (nxdp+2));

    Device_SafeMemoryAlloc(f1Buf.p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryAlloc(f2Buf.p[i], REAL, (nxdp+2)*(ny+2));
    Device_SafeMemoryAlloc(gBuf.p[i], REAL, (nxdp+2)*(ny+2));

  }

  Device_SafeMemoryAlloc(deviceStructF1, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructFeq1, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructF2, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructFeq2, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructG, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructGeq, pop_type, 1);

  Device_SafeMemoryAlloc(deviceStructMemoryup1, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructMemoryup2, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructMemoryupG, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructBndUpG, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructMemorydown1, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructMemorydown2, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructMemorydownG, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructBndDownG, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructF1Buf, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructF2Buf, pop_type, 1);
  Device_SafeMemoryAlloc(deviceStructGBuf, pop_type, 1);

  Device_SafeMemoryCopyToDevice(deviceStructF1, &f1, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructFeq1, &feq1, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructF2, &f2, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructFeq2, &feq2, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructG, &g, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructGeq, &geq, pop_type, 1);

  Device_SafeMemoryCopyToDevice(deviceStructMemoryup1, &memoryup1, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructMemoryup2, &memoryup2, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructMemoryupG, &memoryupG, pop_type, 1);  
  Device_SafeMemoryCopyToDevice(deviceStructBndUpG, &boundaryupG, pop_type, 1);  
  Device_SafeMemoryCopyToDevice(deviceStructMemorydown1, &memorydown1, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructMemorydown2, &memorydown2, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructMemorydownG, &memorydownG, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructBndDownG, &boundarydownG, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructF1Buf, &f1Buf, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructF2Buf, &f2Buf, pop_type, 1);
  Device_SafeMemoryCopyToDevice(deviceStructGBuf, &gBuf, pop_type, 1);

  Device_SafeMemoryAlloc(d_flag, int, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(u1pre, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(v1pre, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(rho1pre, REAL, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(u1post, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(v1post, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(rho1post, REAL, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(u2pre, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(v2pre, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(rho2pre, REAL, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(u2post, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(v2post, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(rho2post, REAL, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(temperaturepre, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(temperaturepost, REAL, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(frcex1, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(frcey1, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(frcex2, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(frcey2, REAL, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(eqfieldx1, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(eqfieldy1, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(eqfieldx2, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(eqfieldy2, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(eqfieldxG, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(eqfieldyG, REAL, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(psi1self, REAL, (nxdp+4)*(ny+2));
  Device_SafeMemoryAlloc(psi2self, REAL, (nxdp+4)*(ny+2));

  Device_SafeMemoryAlloc(utot, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(vtot, REAL, (nxdp+2)*(ny+2));

  Device_SafeMemoryAlloc(tau, REAL, (nxdp+2)*(ny+2));
  Device_SafeMemoryAlloc(tauG, REAL, (nxdp+2)*(ny+2));

  if(myid==0) {
    nxhost=nx;
  } else {
    nxhost=nxdp;
  }

  pxx = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));

  pxy11 = (REAL *) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  pxy12 = (REAL *) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  pxy22 = (REAL *) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));

  pxy = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  pyy = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));

  pxxkin = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  pxykin = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  pyykin = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));

  pop_type hostF1;
  pop_type hostF2;
  pop_type hostG;

  REAL *hostRho1pre;
  REAL *hostRho2pre;
  REAL *hostGpre;

  REAL *hostPsi1self;
  REAL *hostPsi2self;
  REAL *hostUtot;
  REAL *hostVtot;
  REAL *hostFrcex1;
  REAL *hostFrcey1;
  REAL *hostFrcex2;
  REAL *hostFrcey2;

  
  for(i=0; i<npop; i++) {
    hostF1.p[i] = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
    hostF2.p[i] = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
    hostG.p[i] = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  }

  hostRho1pre   = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostRho2pre   = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostGpre      = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostPsi1self  = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostPsi2self  = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostUtot      = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostVtot      = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostFrcex1    = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostFrcey1    = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostFrcex2    = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));
  hostFrcey2    = (REAL*) Malloc((nxhost+2)*(ny+2)*sizeof(REAL));

  flag = (int *) Malloc((nxhost+2)*(ny+2)*sizeof(int));

  flagging(flag,nxhost);

  Device_SafeMemoryCopyToDevice(d_flag, flag, int, (nxdp+2)*(ny+2));

  dellar();
#if defined(USEMPI)
  interexch=SetUpMpi(myid,nprocs,ny+2);


  for(i=0, iarr=0; i<npop; i++) {
      tp[iarr++] = hostF1.p[i]+(ny+2);
  }
  for(i=0; i<npop; i++) {
      tp[iarr++] = hostF2.p[i]+(ny+2);
  }

  for(i=0; i<npop; i++) {
    tp[iarr++] = hostG.p[i]+(ny+2);
  }

  tp[iarr++]=hostRho1pre+(ny+2);
  tp[iarr++]=hostRho2pre+(ny+2);
  tp[iarr++]=hostGpre+(ny+2);

  tp[iarr++]=hostPsi1self+(ny+2);
  tp[iarr++]=hostPsi2self+(ny+2);
  tp[iarr++]=hostUtot+(ny+2);
  tp[iarr++]=hostVtot+(ny+2);
  tp[iarr++]=hostFrcex1+(ny+2);
  tp[iarr++]=hostFrcey1+(ny+2);
  tp[iarr++]=hostFrcex2+(ny+2);
  tp[iarr++]=hostFrcey2+(ny+2);

  if(iarr!=NATB) {
    writelog(TRUE,APPLICATION_RC,"Unexpected value for array index: %d",iarr);
  }
#endif

  stream = (cudaStream_t*) Malloc(ngpu*nstreams * sizeof(cudaStream_t));

  for(int i = 0; i < ngpu; i++) {
    if(ngpu>1) {
      MY_CUDA_CHECK( cudaSetDevice( i ) );
    }
    MY_CUDA_CHECK( cudaStreamCreate(&(stream[i])) );
    MY_CUDA_CHECK( cudaStreamCreate(&(stream[i+ngpu])) );
  }

  if(scratch) {

    if(useTHERMAL){
     
      printf("Check! At preparation step you have velocity perturbation amplitude = %e \n",PERTURB_VELO_AMPLITUDE);
      inithydro_thermal(temperaturepre,initcondG,deviceStructBndUpG,deviceStructBndDownG,Tup,Tdown,myid);
      equili_thermal(temperaturepre,deviceStructGeq);
      initpop(deviceStructG,deviceStructGeq);

    }

    inithydro(u1pre,v1pre,rho1pre,u2pre,v2pre,utot,vtot,rho2pre,initcond,myid,PERTURB_VELO_AMPLITUDE);

    equili(u1pre,v1pre,rho1pre,deviceStructFeq1);
    initpop(deviceStructF1,deviceStructFeq1);

    equili(u2pre,v2pre,rho2pre,deviceStructFeq2);
    initpop(deviceStructF2,deviceStructFeq2);
    
  } else {

    if(useTHERMAL){
      
      if(postPreparation){
      
	restore(f1,"conf1.in",myid);
	restore(f2,"conf2.in",myid);
	
	restore(g,"confG.in",myid);
	
	Device_SafeMemoryAlloc(deviceTemporaneoF1, pop_type, 1);
	Device_SafeMemoryAlloc(deviceTemporaneoF2, pop_type, 1);
	
	Device_SafeMemoryCopyToDevice(deviceTemporaneoF1, &f1, pop_type, 1);
	Device_SafeMemoryCopyToDevice(deviceTemporaneoF2, &f2, pop_type, 1);
	
	rhocomp(rho1pre,deviceTemporaneoF1);
	rhocomp(rho2pre,deviceTemporaneoF2);
	
	initVelocityPerturbationRestart_thermal(u1pre,v1pre,u2pre,v2pre,utot,vtot,PERTURB_VELO_AMPLITUDE);
	inithydro_thermal(temperaturepre,initcondG,deviceStructBndUpG,deviceStructBndDownG,Tup,Tdown,myid);
	
	equili(u1pre,v1pre,rho1pre,deviceStructFeq1);
	initpop(deviceTemporaneoF1,deviceStructFeq1);
	
	equili(u2pre,v2pre,rho2pre,deviceStructFeq2);
	initpop(deviceTemporaneoF2,deviceStructFeq2);
	
	equili_thermal(temperaturepre,deviceStructGeq);
	initpop(deviceStructG,deviceStructGeq);
	
	Device_SafeMemoryCopyFromDevice(&f1, deviceTemporaneoF1, pop_type, 1);
	Device_SafeMemoryCopyFromDevice(&f2, deviceTemporaneoF2, pop_type, 1);
      
      }else{
	printf("You're not in a post-preparation run, so remember to set perturbation amplitude equal to zero! \n");
	inithydro(u1pre,v1pre,rho1pre,u2pre,v2pre,utot,vtot,rho2pre,initcond,myid,PERTURB_VELO_AMPLITUDE);
	inithydro_thermal(temperaturepre,initcondG,deviceStructBndUpG,deviceStructBndDownG,Tup,Tdown,myid);

	restore(f1,"conf1.in",myid);
	restore(f2,"conf2.in",myid);
	restore(g,"confG.in",myid);
	
	inithydro_thermal(temperaturepre,initcondG,deviceStructBndUpG,deviceStructBndDownG,Tup,Tdown,myid);
	  
	equili_thermal(temperaturepre,deviceStructGeq);
	initpop(deviceStructG,deviceStructGeq);
      }

    }else{

      inithydro(u1pre,v1pre,rho1pre,u2pre,v2pre,utot,vtot,rho2pre,initcond,myid,PERTURB_VELO_AMPLITUDE);
      restore(f1,"conf1.in",myid);
      restore(f2,"conf2.in",myid);

    }
  }

  setTau(tau, relax1);

  if(useTHERMAL){
     setTau(tauG, relaxG);
  }

  for(istep=isteprest;istep<=nsteps;istep++){
    TIMER_START;
    if(buffered)
      {
        deviceStructF1Source = deviceStructF1Buf;
        deviceStructF2Source = deviceStructF2Buf;
	deviceStructGSource  = deviceStructGBuf;
        deviceStructF1Dest   = deviceStructF1;
        deviceStructF2Dest   = deviceStructF2;
	deviceStructGDest    = deviceStructG;
	
        pF1Source            = &f1Buf;
        pF2Source            = &f2Buf;
	pGSource             = &gBuf;
        pF1Dest              = &f1;
        pF2Dest              = &f2;
	pGDest               = &g;
	
      }
    else
      {
        deviceStructF1Source = deviceStructF1;
        deviceStructF2Source = deviceStructF2;
	deviceStructGSource  = deviceStructG;
	deviceStructF1Dest   = deviceStructF1Buf;
        deviceStructF2Dest   = deviceStructF2Buf;
	deviceStructGDest    = deviceStructGBuf;
	
        pF1Source            = &f1;
        pF2Source            = &f2;
	pGSource             = &g;
	
        pF1Dest              = &f1Buf;
        pF2Dest              = &f2Buf;
	pGDest               = &gBuf;
      }
    
    buffered = !buffered;
    uwalluptd=uwallup;
    uwalldowntd=uwalldown;
    
    
#if defined(USEMPI)    
    if(ifirst==1 && nprocs>1) {
      ifirst=0;
        ExchangePop(pF1Source, pF2Source, ny+2);
	ExchangePop_thermal(pGSource, ny+2);
        for(i=0; i<ngpu; i++) {
          if(ngpu>1) MY_CUDA_CHECK ( cudaSetDevice( i ) );
          MY_CUDA_CHECK( cudaDeviceSynchronize() );
        }
    }
#endif

    if((pbcy==0) && (roughWallDown==0 && roughWallUp==0)){
      moveplusforcingconstructWW(deviceStructF1Source, deviceStructF1Dest,
                                 deviceStructF2Source, deviceStructF2Dest,
                                 pF1Source, pF1Dest, pF2Source, pF2Dest,
                                 rho1pre,rho2pre,temperaturepre,frcex1,frcey1,frcex2,frcey2,
                                 psi1self+(ny+2),psi2self+(ny+2),
                                 deviceStructMemoryup1,deviceStructMemoryup2,
                                 deviceStructMemorydown1,deviceStructMemorydown2,
                                 uwalluptd,uwalldowntd,utot,vtot,tau);
      
      if(useTHERMAL){
	moveconstructWW_thermal(deviceStructGSource, deviceStructGDest,
				pGSource, pGDest, temperaturepre,deviceStructBndUpG,
				deviceStructBndDownG,Tup,Tdown,initcondG);
      }

    }

    if((pbcy==0) && (roughWallDown==1 || roughWallUp==1)){
      moveplusforcingconstructobstacle(deviceStructF1Source, deviceStructF1Dest,
				       deviceStructF2Source, deviceStructF2Dest,
				       pF1Source, pF1Dest, pF2Source, pF2Dest,
				       rho1pre,rho2pre,temperaturepre,frcex1,frcey1,frcex2,frcey2,
				       psi1self+(ny+2),psi2self+(ny+2),
				       deviceStructMemoryup1,deviceStructMemoryup2,
				       deviceStructMemorydown1,deviceStructMemorydown2,
                                       uwalluptd,uwalldowntd,utot,vtot, d_flag,tau);
      
      if(useTHERMAL){
	moveconstructobstacle_thermal(deviceStructGSource, deviceStructGDest,
				      pGSource, pGDest,temperaturepre,
				      deviceStructBndUpG,deviceStructBndDownG,
				      Tup,Tdown,initcondG, d_flag);
      }
    }
    

    if(pbcy==1){
      moveplusforcingconstructPBC(deviceStructF1Source, deviceStructF1Dest,
                                  deviceStructF2Source, deviceStructF2Dest,
                                  pF1Source, pF1Dest, pF2Source, pF2Dest,
                                  rho1pre,rho2pre,temperaturepre,frcex1,frcey1,frcex2,frcey2,
                                  psi1self+(ny+2),psi2self+(ny+2),tau);
      
      if(useTHERMAL){
	fprintf(stderr,"Thermal with periodic boundary conditions along y is not supported, check input parameters!\n");
	exit(1);
      }
      
    }

    if(pbcy!=0 && pbcy!=1) {
       for(i=0; i<npop; i++) {
         Device_SafeMemoryCopyDevice(pF1Dest->p[i], pF1Source->p[i], REAL,
                                     (nxdp+2)*(ny+2));
         Device_SafeMemoryCopyDevice(pF2Dest->p[i], pF2Source->p[i], REAL,
                                     (nxdp+2)*(ny+2));
	 if(useTHERMAL){
	   fprintf(stderr,"Thermal with boundary condition along y different from is not supported, check input parameters!\n");
	   exit(1);
	 }
       }
    }

    hydrovar(u1pre,v1pre,rho1pre,deviceStructF1Dest,d_flag, rhowall1);
    hydrovar(u2pre,v2pre,rho2pre,deviceStructF2Dest,d_flag, rhowall2);
    
    if(useTHERMAL){
      thermalvar(temperaturepre,deviceStructGDest,d_flag);
    }
    
       
    if(useTHERMAL){
 	 computeTauAndUeq_thermal(frcex1,frcey1,frcex2,frcey2,u1pre,u2pre,v1pre,v2pre,rho1pre,rho2pre,eqfieldxG,eqfieldyG);
    }

    computeTauAndUeq(frcex1,frcey1,frcex2,frcey2,u1pre,u2pre,v1pre,v2pre,rho1pre,rho2pre,eqfieldx1,eqfieldy1,eqfieldx2,eqfieldy2,tau);

    equili(eqfieldx1,eqfieldy1,rho1pre,deviceStructFeq1);

    collis(deviceStructF1Dest,deviceStructFeq1,tau);
    
    equili(eqfieldx2,eqfieldy2,rho2pre,deviceStructFeq2);
    
    collis(deviceStructF2Dest,deviceStructFeq2,tau);
    

    if(useTHERMAL){
      equili(eqfieldxG,eqfieldyG,temperaturepre,deviceStructGeq);
      collis(deviceStructGDest,deviceStructGeq,tauG);
    }

#if defined(USEMPI)    
    if(nprocs>1) {
      SendPop2CPU(pF1Dest, pF2Dest, ny+2);
    }
#endif
    
    hydrovarA(u1post,v1post,rho1post,deviceStructF1Dest,d_flag,rhowall1);
    hydrovarA(u2post,v2post,rho2post,deviceStructF2Dest,d_flag,rhowall2);

    if(useTHERMAL){
      thermalvarA(temperaturepost,deviceStructGDest,d_flag);
    }
#if defined(USEMPI)
    if(nprocs>1) {
      MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));
      if(ngpu>1) {
        MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu-1]));
      }
      interexch->nbyte=2*npop*(ny+2)*sizeof(REAL);
      ExchMpi(interexch);
      RecvPopFromCPU(pF1Dest, pF2Dest, ny+2);
    }
#endif    

    for(i=0; i<ngpu; i++) {
       if(ngpu>1) MY_CUDA_CHECK ( cudaSetDevice( i ) );
       MY_CUDA_CHECK( cudaDeviceSynchronize() );
    }

    AVERAGE(u1pre,v1pre,rho1pre,u2pre,v2pre,rho2pre,u1post,v1post,rho1post,u2post,v2post,rho2post,utot,vtot);

    TIMER_STOP;
    tsimtime+=TIMER_ELAPSED;

    if((istep%nout)==0){
      if(myid==0) {
         printf("%d \n",istep);
      }

      for(i=0; i<npop; i++) {
        Device_SafeMemoryCopyFromDevice(hostF1.p[i],       pF1Dest->p[i], REAL, (nxdp+2)*(ny+2))
        Device_SafeMemoryCopyFromDevice(hostF2.p[i],       pF2Dest->p[i], REAL, (nxdp+2)*(ny+2));
	
	if(useTHERMAL){
	  Device_SafeMemoryCopyFromDevice(hostG.p[i],       pGDest->p[i], REAL, (nxdp+2)*(ny+2));
	}
      }

      Device_SafeMemoryCopyFromDevice(hostRho1pre,  rho1pre, REAL, (nxdp+2)*(ny+2));
      Device_SafeMemoryCopyFromDevice(hostRho2pre,  rho2pre, REAL, (nxdp+2)*(ny+2));

      if(useTHERMAL){
	Device_SafeMemoryCopyFromDevice(hostGpre,  temperaturepre, REAL, (nxdp+2)*(ny+2));
      }

      Device_SafeMemoryCopyFromDevice(hostPsi1self, psi1self+(ny+2),REAL,(nxdp+2)*(ny+2));
      Device_SafeMemoryCopyFromDevice(hostPsi2self, psi2self+(ny+2),REAL,(nxdp+2)*(ny+2));
      Device_SafeMemoryCopyFromDevice(hostUtot,     utot, REAL, (nxdp+2)*(ny+2));
      Device_SafeMemoryCopyFromDevice(hostVtot,     vtot, REAL, (nxdp+2)*(ny+2));
      Device_SafeMemoryCopyFromDevice(hostFrcex1,   frcex1, REAL, (nxdp+2)*(ny+2));
      Device_SafeMemoryCopyFromDevice(hostFrcey1,   frcey1, REAL, (nxdp+2)*(ny+2));
      Device_SafeMemoryCopyFromDevice(hostFrcex2,   frcex2, REAL, (nxdp+2)*(ny+2));
      Device_SafeMemoryCopyFromDevice(hostFrcey2,   frcey2, REAL, (nxdp+2)*(ny+2));
#if defined(USEMPI)
      if(nprocs>1) {
        for(iarr=0; iarr<NATB; iarr++) {
          if(myid>0) {
            int ls;
            ls=(myid<(nprocs-1))?0:(ny+2)*sizeof(REAL);
            if(Gather_send(tp[iarr], (nxdp)*(ny+2)*sizeof(REAL), ls)<0) {
              writelog(TRUE,APPLICATION_RC,"Error in MPI_Gather send");
            }
          } else {
            if(Gather_recv(tp[iarr], (nxdp)*(ny+2)*sizeof(REAL),
                                            (ny+2)*sizeof(REAL))<0) {
              writelog(TRUE,APPLICATION_RC,"Error in MPI_Gather recv");
            }
          }
        }
      }
#endif


      if(myid==0) {
        media(hostRho1pre,hostRho2pre,flag);

        computeptensor(hostF1,hostF2,hostPsi1self,hostPsi2self,hostRho1pre,hostRho2pre,pxx,pxy,pyy,pxxkin,pxykin,pyykin);
        mediatutto(hostF1,hostF2,hostPsi1self,hostPsi2self,hostRho1pre,hostRho2pre,hostUtot,hostVtot,hostFrcex1,hostFrcey1,hostFrcex2,hostFrcey2,pxx,pxy11,pxy12,pxy22,pyy,pxxkin,pxykin,pyykin);
	
      }
#if defined(USEMPI)      
      barrier();
#endif      
    }

    if(dooutenergy){

      if(noutenergy > 0 && (istep%noutenergy)==0){
	
	if(myid==0) {
	  computeEnergy(hostUtot,hostVtot);
	}
      }
#if defined(USEMPI)      
      barrier();
#endif
    }

    if((noutdens > 0 && (istep%noutdens)==0)){
      if(myid==0) {

	if(useTHERMAL){
	  SNAPSHOT_THERMAL(hostGpre);
	}

        SNAPSHOT(hostRho1pre,hostRho2pre,pxy,pxykin);

      }
#if defined(USEMPI)      
      barrier();
#endif
    }
    if((istep%noutconfig)==0) {
      char fileName1[1024];
      snprintf(fileName1, sizeof(fileName1), "conf1_%d.in",
               noutconfigCount%noutconfigMod);

      char fileName2[1024];
      snprintf(fileName2, sizeof(fileName2), "conf2_%d.in",
               noutconfigCount%noutconfigMod);

      char fileNameG[1024];
      if(useTHERMAL){
	snprintf(fileNameG, sizeof(fileNameG), "confG_%d.in",
		 noutconfigCount%noutconfigMod);
      }

      noutconfigCount++;

        if(buffered) {
          dump(f1Buf,fileName1, myid);
          dump(f2Buf,fileName2, myid);

	  if(useTHERMAL){
	    dump(gBuf,fileNameG, myid);
	  }

        } else {
          dump(f1,fileName1,myid);
          dump(f2,fileName2,myid);
	  
	  if(useTHERMAL){
	    dump(g,fileNameG,myid);
	  }
        }
#if defined(USEMPI)      
        barrier();
#endif
    }

    if(ncheckdelaunay > 0 && istep > nmindelaunay
       && istep%ncheckdelaunay == 0){

      dTrigger = delaunayTriggerV4(rho1pre, d_flag, nx, ny, bubbleThreshold,
                                   istep, whichgpu, stream,
                                   pbcx, pbcy, delaunayDebug);
      
      if(dTrigger == 1){
	printf("Flag isomorphism:  %d\n", dTrigger);
      }else{
	printf("Flag isomorphism: 0 \n");
      }
    }


  }

  if(buffered) {
        dump(f1Buf,"conf1.in", myid);
        dump(f2Buf,"conf2.in", myid);
	
	if(useTHERMAL){
	  dump(gBuf,"confG.in", myid);
	}
  } else {
        dump(f1,"conf1.in",myid);
        dump(f2,"conf2.in",myid);
        dump(g,"confG.in",myid);
  }

  for(i=0; i<npop; i++) {
    Device_SafeMemoryFree(f1.p[i]);
    Device_SafeMemoryFree(feq1.p[i]);

    Device_SafeMemoryFree(f2.p[i]);
    Device_SafeMemoryFree(feq2.p[i]);

    Device_SafeMemoryFree(g.p[i]);
    Device_SafeMemoryFree(geq.p[i]);

    Device_SafeMemoryFree(memoryup1.p[i]);
    Device_SafeMemoryFree(memoryup2.p[i]);
    Device_SafeMemoryFree(memoryupG.p[i]);

    Device_SafeMemoryFree(memorydown1.p[i]);
    Device_SafeMemoryFree(memorydown2.p[i]);
    Device_SafeMemoryFree(memorydownG.p[i]);

    Device_SafeMemoryFree(f1Buf.p[i]);
    Device_SafeMemoryFree(f2Buf.p[i]);
    Device_SafeMemoryFree(gBuf.p[i]);
  
  }

  Device_SafeMemoryFree(u1pre);
  Device_SafeMemoryFree(v1pre);
  Device_SafeMemoryFree(rho1pre);

  Device_SafeMemoryFree(u1post);
  Device_SafeMemoryFree(v1post);
  Device_SafeMemoryFree(rho1post);

  Device_SafeMemoryFree(u2pre);
  Device_SafeMemoryFree(v2pre);
  Device_SafeMemoryFree(rho2pre);

  Device_SafeMemoryFree(u2post);
  Device_SafeMemoryFree(v2post);
  Device_SafeMemoryFree(rho2post);

  Device_SafeMemoryFree(temperaturepre);
  Device_SafeMemoryFree(temperaturepost);

  Device_SafeMemoryFree(frcex1);
  Device_SafeMemoryFree(frcey1);
  Device_SafeMemoryFree(frcex2);
  Device_SafeMemoryFree(frcey2);

  Device_SafeMemoryFree(eqfieldx1);
  Device_SafeMemoryFree(eqfieldy1);
  Device_SafeMemoryFree(eqfieldx2);
  Device_SafeMemoryFree(eqfieldy2);
  Device_SafeMemoryFree(eqfieldxG);
  Device_SafeMemoryFree(eqfieldyG);

  Device_SafeMemoryFree(psi1self);
  Device_SafeMemoryFree(psi2self);

  Device_SafeMemoryFree(utot);
  Device_SafeMemoryFree(vtot);

  Device_SafeMemoryFree(tau);
  Device_SafeMemoryFree(tauG);

  Device_SafeMemoryFree(deviceStructF1);
  Device_SafeMemoryFree(deviceStructFeq1);
  Device_SafeMemoryFree(deviceStructF2);
  Device_SafeMemoryFree(deviceStructFeq2);
  Device_SafeMemoryFree(deviceStructG);
  Device_SafeMemoryFree(deviceStructGeq);

  Device_SafeMemoryFree(deviceStructMemoryup1);
  Device_SafeMemoryFree(deviceStructMemoryup2);
  Device_SafeMemoryFree(deviceStructMemoryupG);

  Device_SafeMemoryFree(deviceStructMemorydown1);
  Device_SafeMemoryFree(deviceStructMemorydown2);
  Device_SafeMemoryFree(deviceStructMemorydownG);

  Device_SafeMemoryFree(deviceStructF1Buf);
  Device_SafeMemoryFree(deviceStructF2Buf);
  Device_SafeMemoryFree(deviceStructGBuf);

  Device_SafeMemoryFree(deviceTemporaneoF1);
  Device_SafeMemoryFree(deviceTemporaneoF2);
  

  Free(pxx);
  Free(pxy);
  Free(pyy);

  Free(pxxkin);
  Free(pxykin);
  Free(pyykin);

  for(i=0; i<npop; i++) {
    Free(hostF1.p[i]);
    Free(hostF2.p[i]);
    Free(hostG.p[i]);
  }

  Free(hostRho1pre);
  Free(hostRho2pre);
  Free(hostGpre);
  Free(hostPsi1self);
  Free(hostPsi2self);
  Free(hostUtot);
  Free(hostVtot);
  Free(hostFrcex1);
  Free(hostFrcey1);
  Free(hostFrcex2);
  Free(hostFrcey2);


#if defined(USEMPI)  
  barrier();
#endif
  if(myid==0) {
        printf("Run... done in %f seconds on %d gpus\n",tsimtime*0.000001, nprocs);
  }
#if defined(USEMPI)  
  StopMpi();
#endif
}
#if defined(USEMPI)
struct mpiexch *SetUpMpi(int id, int np, int bndysize) {
  struct mpiexch *p;
  p=(struct mpiexch*)Malloc(sizeof(struct mpiexch));
  MY_CUDA_CHECK( cudaMallocHost((void **)&p->rightsend,bndysize*sizeof(REAL)*npop*2) );
  MY_CUDA_CHECK( cudaMallocHost((void **)&p->rightrecv,bndysize*sizeof(REAL)*npop*2) );
  MY_CUDA_CHECK( cudaMallocHost((void **)&p->leftsend,bndysize*sizeof(REAL)*npop*2) );
  MY_CUDA_CHECK( cudaMallocHost((void **)&p->leftrecv,bndysize*sizeof(REAL)*npop*2) );
  p->right=id+1<np?id+1:0;
  p->left=id-1<0?np-1:id-1;
  p->nbyte=bndysize*sizeof(REAL)*npop*2;
  p->post=FALSE;
  return p;
}

void ExchangeRho(REAL *p1, REAL *p2, int n) {
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend,
                                        p1+n, n*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+n,
                                        p2+n, n*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend,
                                        p1+n*nxdp, n*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+n,
                                        p2+n*nxdp, n*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );

        MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        if(ngpu>1) {
          MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu-1]));
        }
        interexch->nbyte=2*n*sizeof(REAL);
        ExchMpi(interexch);

        MY_CUDA_CHECK( cudaMemcpyAsync(p1,
                                       interexch->leftrecv, n*sizeof(REAL),
                                       cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p2,
                                       interexch->leftrecv+n, n*sizeof(REAL),
                                       cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p1+n*(nxdp+1),
                                       interexch->rightrecv, n*sizeof(REAL),
                                     cudaMemcpyHostToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p2+n*(nxdp+1),
                                       interexch->rightrecv+n, n*sizeof(REAL),
                                     cudaMemcpyHostToDevice, stream[ngpu-1] ) );
       for(int i=0; i<ngpu; i++) {
           if(ngpu>1) MY_CUDA_CHECK ( cudaSetDevice( i ) );
           MY_CUDA_CHECK( cudaDeviceSynchronize() );
       }
}

void ExchangePsi(REAL *p1, REAL *p2, int n) {
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend,
                                        p1+2*n, 2*n*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+2*n,
                                        p2+2*n, 2*n*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend,
                                        p1+n*nxdp, 2*n*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+2*n,
                                        p2+n*nxdp, 2*n*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );

        MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        if(ngpu>1) {
          MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu-1]));
        }
        interexch->nbyte=4*n*sizeof(REAL);
        ExchMpi(interexch);

        MY_CUDA_CHECK( cudaMemcpyAsync(p1,
                                       interexch->leftrecv, 2*n*sizeof(REAL),
                                       cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p2,
                                      interexch->leftrecv+2*n, 2*n*sizeof(REAL),
                                       cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p1+n*(nxdp+2),
                                       interexch->rightrecv, 2*n*sizeof(REAL),
                                     cudaMemcpyHostToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p2+n*(nxdp+2),
                                     interexch->rightrecv+2*n, 2*n*sizeof(REAL),
                                     cudaMemcpyHostToDevice, stream[ngpu-1] ) );
        for(int i=0; i<ngpu; i++) {
           if(ngpu>1) MY_CUDA_CHECK ( cudaSetDevice( i ) );
           MY_CUDA_CHECK( cudaDeviceSynchronize() );
        }
}

 void ExchangePsiRho(REAL *p1, REAL *p2, REAL *p3, REAL *p4, int n) {
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend,
                                        p1+2*n, 2*n*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+2*n,
                                        p2+2*n, 2*n*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+4*n,
                                        p3+n, n*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+5*n,
                                        p4+n, n*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend,
                                        p1+n*nxdp, 2*n*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+2*n,
                                        p2+n*nxdp, 2*n*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+4*n,
                                        p3+n*nxdp, n*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+5*n,
                                        p4+n*nxdp, n*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );

        MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        if(ngpu>1) {
          MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu-1]));
        }
        interexch->nbyte=6*n*sizeof(REAL);
        ExchMpi(interexch);

        MY_CUDA_CHECK( cudaMemcpyAsync(p1,
                                       interexch->leftrecv, 2*n*sizeof(REAL),
                                       cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p2,
                                      interexch->leftrecv+2*n, 2*n*sizeof(REAL),
                                       cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p3,
                                       interexch->leftrecv+4*n, n*sizeof(REAL),
                                       cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p4,
                                       interexch->leftrecv+5*n, n*sizeof(REAL),
                                       cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p1+n*(nxdp+2),
                                       interexch->rightrecv, 2*n*sizeof(REAL),
                                     cudaMemcpyHostToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p2+n*(nxdp+2),
                                     interexch->rightrecv+2*n, 2*n*sizeof(REAL),
                                     cudaMemcpyHostToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p3+n*(nxdp+1),
                                       interexch->rightrecv+4*n, n*sizeof(REAL),
                                     cudaMemcpyHostToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p4+n*(nxdp+1),
                                       interexch->rightrecv+5*n, n*sizeof(REAL),
                                     cudaMemcpyHostToDevice, stream[ngpu-1] ) );

        for(int i=0; i<ngpu; i++) {
           if(ngpu>1) MY_CUDA_CHECK ( cudaSetDevice( i ) );
           MY_CUDA_CHECK( cudaDeviceSynchronize() );
        }

}

void ExchangePop(pop_type *pF1, pop_type *pF2, int n) {
      int i;

      for(i=0; i<npop; i++) {
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+(2*i*(n)),
                                     pF1->p[i]+(n),(n)*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+((2*i+1)*(n)),
                                     pF2->p[i]+(n),(n)*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+(2*i*(n)),
                            pF1->p[i]+((n)*nxdp), (n)*sizeof(REAL),
                                    cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+((2*i+1)*(n)),
                            pF2->p[i]+((n)*nxdp), (n)*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );

      }

      MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));
      if(ngpu>1) {
        MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu-1]));
      }
      interexch->nbyte=2*npop*(n)*sizeof(REAL);
      ExchMpi(interexch);

      for(i=0; i<npop; i++) {
        MY_CUDA_CHECK( cudaMemcpyAsync(pF1->p[i],
                       interexch->leftrecv+(2*i*(n)),
                    (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(pF2->p[i],
                       interexch->leftrecv+((2*i+1)*(n)),
                     (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(pF1->p[i]+((n)*(nxdp+1)),
                       interexch->rightrecv+(2*i*(n)),
                (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(pF2->p[i]+((n)*(nxdp+1)),
                       interexch->rightrecv+((2*i+1)*(n)),
                (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[ngpu-1] ) );
      }

}

void ExchangePop_thermal(pop_type *pF1, int n) {
  int i;

      for(i=0; i<npop; i++) {
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+(2*i*(n)),
                                     pF1->p[i]+(n),(n)*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
	MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+(2*i*(n)),
                            pF1->p[i]+((n)*nxdp), (n)*sizeof(REAL),
                                    cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
      }

      MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));
      if(ngpu>1) {
        MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu-1]));
      }
      interexch->nbyte=2*npop*(n)*sizeof(REAL);
      ExchMpi(interexch);

      for(i=0; i<npop; i++) {
        MY_CUDA_CHECK( cudaMemcpyAsync(pF1->p[i],
                       interexch->leftrecv+(2*i*(n)),
                    (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[0] ) );
      
        MY_CUDA_CHECK( cudaMemcpyAsync(pF1->p[i]+((n)*(nxdp+1)),
				       interexch->rightrecv+(2*i*(n)),
				       (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[ngpu-1] ) );
      }

}

void SendPop2CPU(pop_type *pF1, pop_type *pF2, int n) {
      int i;

      for(i=0; i<npop; i++) {
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+(2*i*(n)),
                                     pF1->p[i]+(n),(n)*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+((2*i+1)*(n)),
                                     pF2->p[i]+(n),(n)*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+(2*i*(n)),
                            pF1->p[i]+((n)*nxdp), (n)*sizeof(REAL),
                                    cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+((2*i+1)*(n)),
                            pF2->p[i]+((n)*nxdp), (n)*sizeof(REAL),
                                     cudaMemcpyDeviceToHost, stream[ngpu-1] ) );

      }
}

void RecvPopFromCPU(pop_type *pF1, pop_type *pF2, int n) {
      int i;
      for(i=0; i<npop; i++) {
        MY_CUDA_CHECK( cudaMemcpyAsync(pF1->p[i],
                       interexch->leftrecv+(2*i*(n)),
                    (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(pF2->p[i],
                       interexch->leftrecv+((2*i+1)*(n)),
                     (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(pF1->p[i]+((n)*(nxdp+1)),
                       interexch->rightrecv+(2*i*(n)),
                (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(pF2->p[i]+((n)*(nxdp+1)),
                       interexch->rightrecv+((2*i+1)*(n)),
                (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[ngpu-1] ) );
      }

}

void SendPop2CPU_thermal(pop_type *pF1, int n) {
      int i;

      for(i=0; i<npop; i++) {
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->leftsend+(2*i*(n)),
                                     pF1->p[i]+(n),(n)*sizeof(REAL),
                                        cudaMemcpyDeviceToHost, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync( interexch->rightsend+(2*i*(n)),
                            pF1->p[i]+((n)*nxdp), (n)*sizeof(REAL),
                                    cudaMemcpyDeviceToHost, stream[ngpu-1] ) );
      }
}

void RecvPopFromCPU_thermal(pop_type *pF1, int n) {
      int i;
      for(i=0; i<npop; i++) {
        MY_CUDA_CHECK( cudaMemcpyAsync(pF1->p[i],
                       interexch->leftrecv+(2*i*(n)),
                    (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[0] ) );

        MY_CUDA_CHECK( cudaMemcpyAsync(pF1->p[i]+((n)*(nxdp+1)),
                       interexch->rightrecv+(2*i*(n)),
                (n)*sizeof(REAL), cudaMemcpyHostToDevice, stream[ngpu-1] ) );
      }

}
#else
void ExchangePsiRho(REAL *p1, REAL *p2, REAL *p3, REAL *p4, int n) {
        MY_CUDA_CHECK( cudaMemcpyAsync(p1+n*(nxdp+2),
                                        p1+2*n, 2*n*sizeof(REAL),
                                        cudaMemcpyDeviceToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p2+n*(nxdp+2),
                                        p2+2*n, 2*n*sizeof(REAL),
                                        cudaMemcpyDeviceToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p3+n*(nxdp+1),
                                        p3+n, n*sizeof(REAL),
                                        cudaMemcpyDeviceToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p4+n*(nxdp+1),
                                        p4+n, n*sizeof(REAL),
                                        cudaMemcpyDeviceToDevice, stream[0] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p1,
                                        p1+n*nxdp, 2*n*sizeof(REAL),
                                     cudaMemcpyDeviceToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p2,
                                        p2+n*nxdp, 2*n*sizeof(REAL),
                                     cudaMemcpyDeviceToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p3,
                                        p3+n*nxdp, n*sizeof(REAL),
                                     cudaMemcpyDeviceToDevice, stream[ngpu-1] ) );
        MY_CUDA_CHECK( cudaMemcpyAsync(p4,
                                        p4+n*nxdp, n*sizeof(REAL),
                                     cudaMemcpyDeviceToDevice, stream[ngpu-1] ) );

        MY_CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        if(ngpu>1) {
          MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu-1]));
        }
}

#endif
