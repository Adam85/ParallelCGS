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
 Content of cgsro.cpp : this function performs a CGS-RO [firstly data is allocated, then computations are performed, loss of orthogonality is calulated and data is free ] 
 Author: Adam Dziekonski
 Generated August 2018
***************************************************************************************/



#include "helpers.h"
#include "cgsro_sequential.h"
#include "cgsro_multicore.h"
#include "cgsro_gpu.h"

void run_cgsro( int m, int n, int ro_steps, int target, double ** A){
    
    printf("A [%d x %d] \n", m, n); 

    // If 1 then the loss orthogonality test if performed
    int performOrthogonalityTest = 1;


    // Arrays to store the times taken by computations in CGS-RO
    double ** timer_seq  = new double*[ro_steps];
    double ** timer_acc  = new double*[ro_steps];
    
    for(int i = 0; i < ro_steps; ++i){
        timer_seq[i]  = new double[9];
        timer_acc[i]  = new double[9];
    }

    double * A_1d = (double*)malloc(sizeof(double)*m*n);
    
    // used in sequential implementation:
    double * Q_1d = (double*)malloc(sizeof(double)*m*n);

    // used in multicore implementation:
    double * Qmulticore_1d;

    // used in gpu implementation:
    double * Qgpu_1d ;
    double * v_1d  ;

    // initialize data for CGSRO implementations:

    // A (2D -> 1D)
    for(int j = 0; j < n; j++){
        for(int i = 0; i < m; i++){
            A_1d[i + j*m] = A[i][j];
        }
    }

    // initialization for sequential 
    for(int j = 0; j < n; j++){
        for(int i = 0; i < m; i++){
            Q_1d[i + j*m]   = 0.0;
        }
    }

    // initialization for multicore
    if (target == 1){
        Qmulticore_1d = (double*)malloc(sizeof(double)*m*n);
        for(int j = 0; j < n; j++){
            for(int i = 0; i < m; i++){
                Qmulticore_1d[i + j*m] = A[i][j];
            }
        }
    } 
    
    // initialization for gpu:
    if (target == 2){
        Qgpu_1d = (double*)malloc(sizeof(double)*m*n);
        v_1d = (double*)malloc(sizeof(double)*m*n);
   
        for(int j = 0; j < n; j++){
            for(int i = 0; i < m; i++){
                Qgpu_1d[i + j*m] = A[i][j];
                v_1d[i+j*m] = A[i][j];
            }
        }
    }

    // GPU warmup: 
    if(target==2){
        double t_warmup = mclock();
        testacc();
        t_warmup = mclock() - t_warmup;
        //printf("GPU warmup = %1.3f [s] \n", t_warmup);
    }

    printf("CGS-RO (reference: sequential on a CPU) :\n"); 
    for (int s = 1; s <= ro_steps; s++){
        
        cgsro_sequential ( A_1d, Q_1d, s, m, n, timer_seq);

        if (performOrthogonalityTest ==1)
            othogonalityTest(Q_1d, m, n, s );
    }


    printf("\nCGS-RO with OPENACC:\n"); 
    double tcgs1 = mclock();
    for (int s = 1; s <= ro_steps; s++){
    
        if (target==1){ // CPU:
            printf("CGS-RO (TARGET=MULTICORE):\n"); 
            
            cgsro_multicore ( A_1d, Qmulticore_1d, s, m, n, timer_acc );
  
            if (performOrthogonalityTest ==1)
                othogonalityTest(Qmulticore_1d, m, n, s );
        }
        if (target==2){ // GPU:
            
            printf("CGS-RO (TARGET=GPU):\n"); 
        
            // it is reqiuired since v_1d is updated in GPU modification and 
            // for new setup (ro_steps) v_1d must be a copy of A      
            for(int j = 0; j < n; j++){
                for(int i = 0; i < m; i++){
                    Qgpu_1d[i + j*m] = A[i][j];
                    v_1d[i+j*m] = A[i][j];
                }
            }
            
            cgsro_gpu  ( Qgpu_1d, v_1d, s, m, n, timer_acc );


            if (performOrthogonalityTest ==1)
                othogonalityTest(Qgpu_1d, m, n, s );

        }
        
    }

    for (int s = 1; s <= ro_steps; s++){    
        printf("Speedup [CGS-RO][# re-orthogonalizations = %2d] = %1.2f \n", s, timer_seq[s-1][8]/timer_acc[s-1][8] );
    }

    printf("\n[-------------------]\n");
    printf("A [%d x %d] \n", m, n); 
    printf("[-------------------]\n");
    for (int s = 0; s < ro_steps; s++){
        printf("[SPEEDUP][No. of re-orthogonalizations: %2d]\n", s);
        printf("[CGS-RO] 1. init       = %1.1f  \n",    timer_seq[s][0]/timer_acc[s][0]);
        printf("[CGS-RO] 2. aj         = %1.1f  \n",    timer_seq[s][1]/timer_acc[s][1]);
        if (target==1)
            printf("[CGS-RO] 3. vj         = %1.1f  \n",    timer_seq[s][2]/timer_acc[s][2] );
        else
            printf("[CGS-RO] 3. vj         = ---  \n") ;
        printf("[CGS-RO] 4. re-ortho   = %1.1f  \n",    (timer_seq[s][3]+timer_seq[s][4]+timer_seq[s][5])/(timer_acc[s][3]+timer_acc[s][4]+timer_acc[s][5]) );
        printf("[CGS-RO]  re-ortho(1)   = %1.1f  \n",   timer_seq[s][3]/timer_acc[s][3] );
        printf("[CGS-RO]  re-ortho(2)   = %1.1f  \n",   timer_seq[s][4]/timer_acc[s][4] );
        printf("[CGS-RO]  re-ortho(3)   = %1.1f  \n",   timer_seq[s][5]/timer_acc[s][5] );
        printf("[CGS-RO] 5. Q          = %1.1f  \n",    timer_seq[s][6]/timer_acc[s][6] );
        printf("[CGS-RO] 1-5 ALL       = %1.2f  \n",    timer_seq[s][8]/timer_acc[s][8] );

        printf("[-------------------]\n");
    }


    delete [] A ;

}

