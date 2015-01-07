#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
////#include <gsl/gsl_rng.h>
////#include <gsl/gsl_randist.h>
#include <fftw3.h>
#include "fitsio.h"
#include "psim.h"

int idft2d (acfStruct *acfStructure);
int dft2d (acfStruct *acfStructure, fftw_complex *out);
int calACF (acfStruct *acfStructure);
int power (acfStruct *acfStructure);
void deallocateMemory (acfStruct *acfStructure);
void allocateMemory (acfStruct *acfStructure);
int simDynSpec (acfStruct *acfStructure);
int calculateScintScale (acfStruct *acfStructure, controlStruct *control);
void preAllocateMemory (acfStruct *acfStructure);
double find_peak_value (int n, double *s);
int calSize (acfStruct *acfStructure, double *size, double *ratio);
int windowSize (acfStruct *acfStructure, double *size);
