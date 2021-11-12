/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

#include "util.h"

char *LogFileName;

void writelog(int end, int rc, const char *fmt, ...) {
  static FILE *filelog=NULL;
  char buf[MAXSTRLEN+1];
  va_list ap;
  va_start(ap, fmt);
#ifdef  HAVE_VSNPRINTF
  vsnprintf(buf, MAXSTRLEN, fmt, ap);
#else
  vsprintf(buf, fmt, ap);
#endif
  if(filelog==NULL) {
    if(LogFileName==NULL || (filelog=fopen(LogFileName,"w"))==NULL) {
                filelog=stderr;
    }
  }
  fputs(buf,filelog);
  if(end) {
    fflush(filelog);
    fclose(filelog);
    exit(rc);
  }
}

void *Malloc(int size) {
  void *r;
  if(size<=0) {
    writelog(TRUE,MALLOC_RC,"malloc invalid size");
  }
  if((r=malloc(size))==NULL) {
    writelog(TRUE,MALLOC_RC,"malloc failed");
  }
  memset(r, 0, size);
  return r;
}

void Free(void *p) {
  free(p);
  p = NULL;
}

FILE *Fopen(const char *filename, const char *mode) {
        FILE    *fp;

        if ( (fp = fopen(filename, mode)) == NULL) {
	  perror("fopen: ");
          writelog(TRUE,FOPEN_RC," failed fopen for %s, mode %s", filename, mode);
	}

        return(fp);
}

char *Strdup(char *p) {
  void *r;
  if(p==NULL) {
    writelog(TRUE,STRDUP_RC,"strdup invalid pointer");
  }
  if((r=strdup(p))==NULL) {
    writelog(TRUE,STRDUP_RC,"strdup");
  }
  return (char*)r;
}
