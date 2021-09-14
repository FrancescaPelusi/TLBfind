/*-------------------------------------------------------------------------*/
/**
   Copyright 2021 Francesca Pelusi, Matteo Lulli, Mauro Sbragaglia and Massimo Bernaschi

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*--------------------------------------------------------------------------*/

#ifndef _INIPARSER_H_
#define _INIPARSER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dictionary.h"

#define iniparser_getstr(d, k)  iniparser_getstring(d, k, NULL)
#define iniparser_setstr        iniparser_setstring

int iniparser_getnsec(dictionary * d);

char * iniparser_getsecname(dictionary * d, int n);

void iniparser_dump_ini(dictionary * d, FILE * f);

void iniparser_dump(dictionary * d, FILE * f);

char * iniparser_getstring(dictionary * d, const char * key, char * def);

#if defined(USEPLAINC)
int iniparser_getint(dictionary * d, const char * key, int notfound);
#else
extern "C" int iniparser_getint(dictionary * d, const char * key, int notfound);
#endif

#if defined(USEPLAINC)
double iniparser_getdouble(dictionary * d, char * key, double notfound);
#else
extern "C" double iniparser_getdouble(dictionary * d, char * key, double notfound);
#endif

int iniparser_getboolean(dictionary * d, const char * key, int notfound);

int iniparser_setstring(dictionary * ini, char * entry, char * val);

void iniparser_unset(dictionary * ini, char * entry);

int iniparser_find_entry(dictionary * ini, char * entry) ;

#if defined(USEPLAINC)
dictionary * iniparser_load(const char * ininame);
#else
extern "C" dictionary * iniparser_load(const char * ininame);
#endif

void iniparser_freedict(dictionary * d);

#endif
