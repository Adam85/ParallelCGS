/*
 Copyright (c) 2018 Gdańsk University of Technology
 
 Unless otherwise indicated, Source Code is licensed under MIT license.
 See further explanation attached in License Statement (distributed in the file LICENSE).
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

/***************************************************************************************
 ParallelCGS: classical Gram-Schmidt with re-orthogonalization:
 Content OF helpers.cpp: helping functions to calc time, alocate matrix, init matrix with random values etc.  
 Author: Adam Dziekonski
 Generated August 2018
***************************************************************************************/

#include "helpers.h"
double mclock(){
    struct timeval tp;
    double sec,
           usec;

    gettimeofday( &tp, NULL );
    sec    = double( tp.tv_sec );
    usec   = double( tp.tv_usec )/1E6;
    return sec + usec;
}

double ** allocMatrix (  int m, int n) {

    double ** A = new double*[m];
    for(int i = 0; i < m; ++i)
        A[i] = new double[n];
    
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            A[i][j] = 0.0;
        }
    }

    return A;

}

void printMatrix( double ** A, int m, int n){
    printf("\n");
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

void initA_version1( double ** A, int m, int n, double epsilon){
    for (int j = 0; j < n; j++){
        A[0][j]   = 1.0;
        if (m != n)
            A[j+1][j] = epsilon;
        else
            A[j][j] = epsilon;
    }
}

void initA_version2( double ** A, int m, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            A[i][j] = 0.1 * rand()/rand();
        }
    }
}

void initI_1d( double * I_1d, int m, int n ){

    for (int j = 0; j < n*n; j++){
        I_1d[j] = 0.0;
    }

    for (int j = 0; j < n; j++){
        I_1d[j*n+j] = 1.0;
    }
}
double normEq2( double ** A, int m, int n){

    double nr = 0.0;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            nr += A[i][j] * A[i][j];
        }
    }

    return sqrt(nr);
}
double normEq2( double * a, int m){

    double nr = 0.0;
    for (int i = 0; i < m; i++){
        nr += a[i] * a[i];
    }

    return sqrt(nr);
}

double normInf_1d( double * A, int m, int n){

    double nr = 0.0;
    double max = -1.0;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            nr = abs(A[i*m+j]) ;
            if (nr > max)
                max = nr;
        }
    }

    return max;
}

void checkLossOfOrthogonality ( double * I_QtQ,  double * QtQ, double * I,  double * Q, int m, int n){

    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            for(int k = 0; k < m; ++k) {
                QtQ[i*n+j] += Q[k+i*m] * Q[k+j*m];
            }

    for (int i = 0; i < n ; i++){
        for (int j = 0; j < n ; j++){
            I_QtQ[i*n+j] = I[i*n+j] - QtQ[i*n+j];
        }
    }
}


void othogonalityTest(double * Q_1d, int m, int n, int s ){

    double time_orthotest = mclock();

    double t1 = mclock();
    
    double * I_1d       = (double*)malloc(sizeof(double)*n*n); // square matrix: I
    double * QtQ_1d     = (double*)malloc(sizeof(double)*n*n); // square matrix: Q^T*Q
    double * I_QtQ_1d   = (double*)malloc(sizeof(double)*n*n); // square matrix: I - Q^T*Q
    
    double t2 = mclock();

    initI_1d(I_1d, n, n);
        
    double t3 = mclock();

    checkLossOfOrthogonality (I_QtQ_1d, QtQ_1d, I_1d, Q_1d, m, n);
    
    double t4 = mclock();

    double norm = normInf_1d(I_QtQ_1d, n, n);

    double t5 = mclock();
    
    time_orthotest = mclock() - time_orthotest;

    printf("[CGS-RO][re-ortho #%d] NormInf(I-Q^T*Q) = %1.3e [TIME of LossOrthogonalityTest: %1.3f s]\n", s, norm , time_orthotest);
    //printf("alloc = %2.2f s, init = %2.2f s, calc I-Q^T*Q = %2.2f s, calc normInf = %2.2f s\n", t2-t1, t3-t2, t4-t3, t5-t4 );
}


// Important: If needed computations from this function may also be parallelized with OpenACC
void checkLossOfOrthogonality ( double ** I_QtQ,  double ** QtQ, double ** I,  double ** Q, int m, int n){
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            for(int k = 0; k < m; ++k) {
                QtQ[i][j] += Q[k][i] * Q[k][j];
            }

    for (int i = 0; i < n ; i++){
        for (int j = 0; j < n ; j++){
            I_QtQ[i][j] = I[i][j] - QtQ[i][j];
        }
    }
}






