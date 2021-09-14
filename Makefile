CC=c++
CCUTIL=cc
LD=nvcc
CFLAGS= -c -O
LDFLAGS= -lm
CUDACC=nvcc
MODERNGPUDIR=minimal_moderngpu/src
MODERNGPUINCLUDE=minimal_moderngpu/include
CUDACFLAGS= -c -O3 -g -m64 -arch=${ARCH} -I${MODERNGPUINCLUDE}
CUDASDKDIR= /usr/local/cuda
ARCH=sm_35
ifndef ARCH
$(error ARCH not set)
endif

all: cudatlbfind buildObstacle dropletStats deltaAnalysis nusseltNumber nusseltNumberDroplet

ifndef ARCH
	$(error ARCH is not set)
endif

cudatlbfind: cudatlbfind.o iniparser.o dictionary.o util.o delaunayCuda.o mgpucontext.o mgpuutil.o
	${LD} -o cudatlbfind cudatlbfind.o iniparser.o dictionary.o util.o mgpucontext.o mgpuutil.o delaunayCuda.o ${LDFLAGS}

buildObstacle: buildObstacle.o 
	${CC} -o buildObstacle buildObstacle.o iniparser.o dictionary.o util.o ${LDFLAGS}

dropletStats: dropletStats.o delaunayCuda.o  mgpucontext.o mgpuutil.o
	${LD} -o dropletStats dropletStats.o delaunayCuda.o mgpucontext.o mgpuutil.o util.o ${LDFLAGS}

deltaAnalysis: deltaAnalysis.o delaunayCuda.o mgpucontext.o mgpuutil.o
	${LD} -o deltaAnalysis deltaAnalysis.o delaunayCuda.o mgpucontext.o  mgpuutil.o util.o ${LDFLAGS}

nusseltNumber: nusseltNumber.o 
	${CC} -o nusseltNumber nusseltNumber.o util.o ${LDFLAGS}

nusseltNumberDroplet: nusseltNumberDroplet.o delaunayCuda.o mgpucontext.o mgpuutil.o
	${LD} -o nusseltNumberDroplet nusseltNumberDroplet.o delaunayCuda.o mgpucontext.o  mgpuutil.o util.o ${LDFLAGS}

.c.o:; $(CC) $(CFLAGS) $< -o $@

%.o: %.cu
	$(CUDACC) $(CUDACFLAGS) -I$(MODERNGPUINCLUDE) $< 

mgpucontext.o:
	nvcc -c -arch=${ARCH} ${MODERNGPUDIR}/mgpucontext.cu -I ${MODERNGPUDIR}/../include/
mgpuutil.o:
	nvcc -c -arch=${ARCH} ${MODERNGPUDIR}/mgpuutil.cpp -I ${MODERNGPUDIR}/../include/
util.o: util.c
	$(CCUTIL) $(CFLAGS) util.c

buildObstacle.o: buildObstacle.c
	$(CCUTIL) $(CFLAGS) -DUSEPLAINC -std=c99 buildObstacle.c

clean:
	rm -rf *.o cudatlbfind buildObstacle dropletStats deltaAnalysis nusseltNumber nusseltNumberDroplet
