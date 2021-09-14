/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#include <mpi.h>
#include "util.h"
#include "tlbfind2dmpi.h"

void barrier(void) {
        MPI_Barrier(MPI_COMM_WORLD);
}

void StartMpi(int *id, int *np, int *argc, char ***argv) {
  int myid, nprocs;
  MPI_Init(argc,argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  *id=myid;
  *np=nprocs;
}

void ExchMpi(struct mpiexch *p) {
            int rc;
            MPI_Status mpi_status[DIRECTIONS];
            static MPI_Request mpireq[DIRECTIONS];
            static int posted=FALSE;
            if(!posted) {
               rc=MPI_Irecv(p->rightrecv, p->nbyte, MPI_BYTE, p->right, LEFTTAG,
                           MPI_COMM_WORLD,&mpireq[0]);
               rc=MPI_Irecv(p->leftrecv, p->nbyte, MPI_BYTE, p->left, RIGHTTAG,
                           MPI_COMM_WORLD,&mpireq[1]);
               posted=TRUE;
            }
            rc=MPI_Send(p->rightsend, p->nbyte, MPI_BYTE, p->right, RIGHTTAG, MPI_COMM_WORLD);
            rc=MPI_Send(p->leftsend, p->nbyte, MPI_BYTE, p->left, LEFTTAG, MPI_COMM_WORLD);
            if(posted) {
              rc=MPI_Waitall(DIRECTIONS,mpireq,mpi_status);
            }
            if(p->post) {
              rc=MPI_Irecv(p->rightrecv, p->nbyte, MPI_BYTE, p->right, RIGHTTAG,
                         MPI_COMM_WORLD,&mpireq[0]);
              rc=MPI_Irecv(p->leftrecv, p->nbyte, MPI_BYTE, p->right, RIGHTTAG,
                         MPI_COMM_WORLD,&mpireq[1]);
              posted=TRUE;
            } else {
              posted=FALSE;
            }
}

MYREAL GlobalSum(MYREAL e, int myid) {
  MYREAL i=e, o;
  int rc;
  if(sizeof(MYREAL)==sizeof(float)) {
    rc=MPI_Reduce(&i, &o, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    rc=MPI_Reduce(&i, &o, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if(myid==0) {
    return o;
  } else {
    return 0.;
  }
}

double MaxTime(double t) {
  double i=t, o;
  int rc;
  rc=MPI_Allreduce(&i, &o, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return o;
}

int Gather_send(void *f, int n, int ls) {
  int np, i;
  static int *displs=NULL;
  static int *counts=NULL;
  if(displs==NULL) {
  	MPI_Comm_size(MPI_COMM_WORLD,&np);
	displs=(int *)Malloc(np*sizeof(int));
	counts=(int *)Malloc(np*sizeof(int));
	for(i=0; i<np-1; i++) {
		counts[i]=n;
		displs[i]=i*n;
	}
	counts[np-1]=n+ls;
	displs[np-1]=(np-1)*n;
  }
  return MPI_Gatherv(f, n+ls, MPI_BYTE, f, counts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);
}

int Gather_recv(void *f, int n, int ls) {
  int np, i;
  static int *displs=NULL;
  static int *counts=NULL;
  if(displs==NULL) {
  	MPI_Comm_size(MPI_COMM_WORLD,&np);
	displs=(int *)Malloc(np*sizeof(int));
	counts=(int *)Malloc(np*sizeof(int));
	for(i=0; i<np-1; i++) {
		counts[i]=n;
		displs[i]=i*n;
	}
	counts[np-1]=n+ls;
	displs[np-1]=(np-1)*n;
  }
  return MPI_Gatherv(MPI_IN_PLACE, n, MPI_BYTE, f, counts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);
}

int Scatter_send(void *f, int n) {
  return MPI_Scatter(f, n, MPI_BYTE, MPI_IN_PLACE, n, MPI_BYTE, 0, MPI_COMM_WORLD);
}

int Scatter_recv(void *f, int n) {
  return MPI_Scatter(f, n, MPI_BYTE, f, n, MPI_BYTE, 0, MPI_COMM_WORLD);
}



void StopMpi() {
  int rc=MPI_Finalize();
}
