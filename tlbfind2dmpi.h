/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

enum {RIGHTTAG=1001, LEFTTAG=1003, INITTAG=1005};
#define DIRECTIONS 2

struct mpiexch {
  MYREAL *rightsend;
  MYREAL *rightrecv;
  MYREAL *leftsend;
  MYREAL *leftrecv;
  int  right;
  int  left;
  int  nbyte;
  int  post;
};

int Gather_send(void *, int, int );
int Gather_recv(void *, int, int );
int Scatter_send(void *, int );
int Scatter_recv(void *, int );
