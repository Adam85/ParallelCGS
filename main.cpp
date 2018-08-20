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
 Content of main.cpp : a main function which runs a test based on the execution setup 
 Author: Adam Dziekonski
 Generated August 2018
***************************************************************************************/

#include "helpers.h"

#include "cgsro.h"

int main( int argc, char* argv[]  ){
    printf( "\n\n\nParallelCGS: classical Gram-Schmidt with re-orthogonalization:\n" );

    // CGS setup:
    // m - number of rows
    // n - number of columns
    // ro_steps - number of reorthogonalization steps
    // target - device which performs computations: 1 - CPU, 2 - GPU
    int m, n, ro_steps, target;
   
    // default:
    m = 1000;
    n = 100;
    ro_steps  = 1;
    target = 2;

    // defined by user:
    m = (int)strtol( argv[1], NULL, 10 );   
    n = (int)strtol( argv[2], NULL, 10 );  
    ro_steps  = (int)strtol( argv[3], NULL, 10 );  
    target = (int)strtol( argv[4], NULL, 10 );  

    printf("CGS setup >>> m(rows) = %d, n(cols) = %d, ro_steps = %d, target = %d\n", m, n, ro_steps, target);

    double ** A = allocMatrix ( m, n) ; // m x n

    // Matrix type #1
    double epsilon = 1e-3;
    initA_version1(A, m, n, epsilon);

    // Matrix type #2
    //initA_version2(A, m, n);

    //printMatrix( A, m, n);

    // run classical Gram-Schmidt with re-orthogonalization due to the execution setup:
    run_cgsro(m,n,ro_steps,target,A); 
    
    return 0;
}
