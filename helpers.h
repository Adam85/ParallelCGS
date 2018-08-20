#include "stdio.h"
#include "stdlib.h"

#include "math.h"
#include "sys/time.h"

#include "/opt/pgi/linux86-64/2017/cuda/9.0/include/cuda.h"
#include "/opt/pgi/linux86-64/2017/cuda/9.0/include/cuComplex.h"

double mclock();

double ** allocMatrix (  int m, int n);

void printMatrix( double ** A, int m, int n);

void initA_version1( double ** A, int m, int n, double epsilon);
void initA_version2( double ** A, int m, int n);

void initI_1d( double * I, int m, int n );

double normEq2( double ** A, int m, int n);
double normEq2( double * a, int m);

double normInf_1d( double * A, int m, int n);

void othogonalityTest(double * , int , int , int );

void checkLossOfOrthogonality ( double * I_QtQ,  double * QtQ, double * I,  double * Q, int m, int n);




