/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#ifndef __ASSERT_H__
#define __ASSERT_H__

#define HostAssert(condition)                                   {                                                                                                                               \
                                                                    if(!(condition))                                                                                                            \
                                                                    {                                                                                                                           \
                                                                        printf("Cuda  assert failed at line %d in file %s:\n%s is not true.\n", __LINE__, __FILE__, #condition);             \
                                                                        exit(1);                                                                                                                \
                                                                    }                                                                                                                           \
                                                                }

#define HostAssertWithMessage(condition, msg)                   {                                                                                                                               \
                                                                    if(!(condition))                                                                                                            \
                                                                    {                                                                                                                           \
                                                                        printf("Cuda  assert failed at line %d in file %s:\n%s (%s is not true).\n", __LINE__, __FILE__, msg, #condition);   \
                                                                        exit(1);                                                                                                                \
                                                                    }                                                                                                                           \
                                                                }
                                                
                                                                                                                                                                                                                                                                                \
#define HostAssertNotNull(arg)                                  HostAssert(pInstance, ((arg) != NULL))
#define HostAssertNotNullWithMessage(arg, msg)                  HostAssertWithMessage(pInstance, ((arg) != NULL), msg)

#define HostAssertParameterNotNull(param)                       HostAssertWithMessage(pInstance, ((param) != NULL), "Internal error")

#endif

