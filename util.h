/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#if!defined(TRUE)
enum {FALSE, TRUE};
#endif
#define SKIPBLANK         while(po[0] & (po[0]==' ' || po[0]=='\t')) po++; \
                          if (!po[0]) { \
                             i++;      \
                             po=argv[i]; \
                          }
enum RC{OK, APPLICATION_RC, STRDUP_RC, MALLOC_RC, FOPEN_RC, POPEN_RC, FGETS_RC,
        SOCKET_RC, SEND_RC, RECV_RC, LOCK_RC, UNLOCK_RC, MAKEMATR_RC,
        MAKEVECT_RC};

#define MAXSTRLEN 1024
#define MAXFILENAME 1024
#define INVALID_INT -1
#define INVALID_REAL -1.e7

#define READINTFI(v,s) snprintf(key,sizeof(key),"%s:%s","INPUTDATA",(s)); \
                          (v)=iniparser_getint(ini, key, INVALID_INT); \
                          if((v)==INVALID_INT) { \
                            writelog(TRUE,APPLICATION_RC, \
                                     "Invalid value for key <%s> from input file %s\n", \
                                     key, inputfile); \
                          }

#define READREALFI(v,s)  snprintf(key,sizeof(key),"%s:%s","INPUTDATA",(s)); \
                            (v)=(REAL)iniparser_getdouble(ini, key, INVALID_REAL);\
                            if((v)==(REAL)INVALID_REAL) { \
                              writelog(TRUE,APPLICATION_RC,\
                                       "Invalid value for key <%s> from input file %s\n",\
                                       key, inputfile); \
                            }

#define CUDAREADINTFI(v,s) snprintf(key,sizeof(key),"%s:%s","CUDA",(s)); \
                          (v)=(unsigned int)iniparser_getint(ini, key, INVALID_INT); \
                          if((int)(v)==INVALID_INT) { \
                            writelog(TRUE,APPLICATION_RC, \
                                     "Invalid value for key <%s> from input file %s\n", \
                                     key, inputfile); \
                          }
