/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#ifndef __DEVICE_CUDA_MEMORY_H__
#define __DEVICE_CUDA_MEMORY_H__

#include "device_cuda.h"

#define Device_SafeMemoryAlloc(ptr, type, size)                DeviceCudaSafeCallNoSynch(cudaMalloc((void **)(&(ptr)), (size)*sizeof(type)))  
                                                                    
#define Device_SafeMemoryFree(ptr)                             if((ptr) != NULL)                                                                       \
                                                                             {                                                                                       \
                                                                                DeviceCudaSafeCallNoSynch(cudaFree((void *)(ptr)));                    \
                                                                             }                                                                  

#define Device_SafeMemoryCopyToDevice(dst, src, type, size)    DeviceCudaSafeCallNoSynch(cudaMemcpy(dst, src, sizeof(type)*(size), cudaMemcpyHostToDevice))
#define Device_SafeMemoryCopyFromDevice(dst, src, type, size)  DeviceCudaSafeCallNoSynch(cudaMemcpy(dst, src, sizeof(type)*(size), cudaMemcpyDeviceToHost))

#define Device_SafeMemoryCopyDevice(dst, src, type, size)      DeviceCudaSafeCallNoSynch(cudaMemcpy(dst, src, sizeof(type)*(size), cudaMemcpyDeviceToDevice))


#define DeviceEnableLoadUInt32ValuesForBlock()                            HostUInt32   __uint32_val;
#define DeviceEnableLoadInt32ValuesForBlock()                             HostInt32    __int32_val;
#define DeviceEnableLoadREALValuesForBlock()                           HostFloat32  __float32_val;


#define Device_SafeLoadUInt32ValueOnDevice(var, value)         __uint32_val = (value);                                                                                                                                 \
                                                                             DeviceCudaSafeCallNoSynch(cudaMemcpyToSymbol(var, &(__uint32_val), sizeof(HostUInt32), 0, cudaMemcpyHostToDevice))
                                                                            
#define Device_SafeLoadInt32ValueOnDevice(var, value)         __int32_val = (value);                                                                                                                                 \
                                                                             DeviceCudaSafeCallNoSynch(cudaMemcpyToSymbol(var, &(__int32_val), sizeof(HostInt32), 0, cudaMemcpyHostToDevice))
                                                                            


#define Device_SafeLoadREALValueOnDevice(var,value)         __float32_val = (value);                                                                                                                                \
                                                                             DeviceCudaSafeCallNoSynch(cudaMemcpyToSymbol(var, &(__float32_val), sizeof(HostFloat32), 0, cudaMemcpyHostToDevice))
                                                                    
#define Device_SafeLoadUInt32ValueFromDevice(var, value)       DeviceCudaSafeCallNoSynch(cudaMemcpyFromSymbol(&(__uint32_val), var, sizeof(HostUInt32), 0, cudaMemcpyDeviceToHost));                 \
                                                                             (value) = __uint32_val

#define Device_SafeLoadInt32ValueFromDevice(var, value)       DeviceCudaSafeCallNoSynch(cudaMemcpyFromSymbol(&(__int32_val), var, sizeof(HostInt32), 0, cudaMemcpyDeviceToHost));                 \
                                                                             (value) = __int32_val
  
  
#define Device_SafeLoadFloat32ValueFromDevice(var,value)       DeviceCudaSafeCallNoSynch(cudaMemcpyFromSymbol(&(__float32_val), var, sizeof(HostFloat32), 0, cudaMemcpyDeviceToHost));               \
                                                                             (value) = __float32_val

#endif

