/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#ifndef __DEVICE_CUDA_H__
#define __DEVICE_CUDA_H__

#include <cuda.h>
#include "host_assert.h"
#include "host_types.h"

#define Device_ParallelRepeatStartingFromAndUntilIndexIsLessThan(arg1, arg2) {                                                                                      \
                                                                                    HostInt32 index;                                                                \
                                                                                    const HostInt32 num_threads = (HostInt32)gridDim.x*(HostInt32)blockDim.x;       \
                                                                                    const HostInt32 istart = (arg1) + (HostInt32)blockDim.x*(HostInt32)blockIdx.x;  \
                                                                                                                                                                    \
                                                                                    for(index = istart+threadIdx.x; index < (arg2); index += num_threads)           \
                                                                                    {


#define Device_ParallelRepeatUntilIndexIsLessThan(arg)                       {                                                                                   \
                                                                                    HostUInt32 index;                                                            \
                                                                                    const HostUInt32 num_threads = gridDim.x*blockDim.x;                         \
                                                                                    const HostUInt32 istart = blockDim.x*blockIdx.x;                             \
                                                                                                                                                                 \
                                                                                    for(index = istart+threadIdx.x; index < (arg); index += num_threads)         \
                                                                                    {

#define Device_ParallelEndRepeat                                                    }                                                                            \
                                                                             }


#define Device_QualKernel                                                    __global__

#define Device_DeclareKernelNoParams(ker)                                    __global__ void ker()

#define Device_DeclareKernel(ker, params)                                    __global__ void ker(params)


#define Device_Synchronize()                                                 cudaDeviceSynchronize()

#define Device_ExecuteKernel(ker)                                            ker<<< sBlocksNum, sThreadsNum >>>

#define Device_ExecuteKernelAS(ker,streamid)                                ker<<< sBlocksNum, sThreadsNum, 0, stream[(streamid)] >>>

#define Device_ExecuteSingleKernel(ker)                                    ker<<< 1, 1 >>>

#define Device_ExecuteKernelNoParams(ker)                                    ker<<< sBlocksNum, sThreadsNum >>>()


#define DeviceCudaSafeCallNoSynch(call)                           {                                                                                       \
                                                                                    cudaError err = call;                                                               \
                                                                                    if(cudaSuccess != err)                                                              \
                                                                                    {                                                                                   \
                                                                                        char msg[2048];                                                                 \
                                                                                        sprintf(msg, "Cuda error in file '%s' in line %i : %s.\n",                      \
                                                                                                __FILE__, __LINE__, cudaGetErrorString( err) );                         \
                                                                                        HostAssertWithMessage(cudaSuccess != err, msg);                   \
                                                                                    }                                                                                   \
                                                                                }

#if __DEVICE_EMULATION__
#define Device_InitDevice()
#else

#define Device_InitDevice(whichgpu)                                                 {                                                                                               \
                                                                                    HostInt32 dev=0;                                                                           \
                                                                                    HostInt32 deviceCount;                                                                   \
    if((int)(whichgpu)>=0) { \
        dev=(whichgpu); \
    } \
    DeviceCudaSafeCallNoSynch(cudaSetDevice(dev)); \
                                                                                    DeviceCudaSafeCallNoSynch(cudaGetDeviceCount(&deviceCount));                  \
                                                                                    HostAssertWithMessage(deviceCount > 0, "No CUDA device available.");          \
                                                                                    for (dev = 0; dev < deviceCount; ++dev)                                                     \
                                                                                    {                                                                                           \
                                                                                        cudaDeviceProp deviceProp;                                                              \
                                                                                        DeviceCudaSafeCallNoSynch(cudaGetDeviceProperties(&deviceProp, dev));     \
                                                                                        if (deviceProp.major >= 1)                                                              \
                                                                                            break;                                                                              \
                                                                                    }                                                                                           \
                                                                                    HostAssertWithMessage(dev < deviceCount, "No CUDA 1.x device available.");    \
   }
#endif


#define DeviceQualConstant   __device__ __constant__
#define DeviceQual           __device__



#define _DEVICE_PADDING_SIZE     16

#include "cuda_memory.h"

#endif

