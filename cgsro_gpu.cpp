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
 Content of cgsro_gpu.cpp : this function incudes an OpenACC implementation for a GPU 
 Author: Adam Dziekonski
 Generated August 2018
***************************************************************************************/

#include "helpers.h"
#include "cgsro_gpu.h"

// Explanation: gclock() and mclock() are sibling functions to measure the time taken by a current step (computations, allocations).
//              However, if here mclock() (defined in helpers.cpp) was called insteed of gclock(), then the following error occurs:
//              'PGCC-W-0155-Invalid accelerator data region: branching into or out of region is not allowed'
//              Defining gclock() in the same file where is the function with data region in which gclock() is used omitts this error.
//              For sure there is general solution to overcome it, but for this not complicated program this not optimal solution was used.

 
double gclock(){
    struct timeval tp;
    double sec,
           usec;

    gettimeofday( &tp, NULL );
    sec    = double( tp.tv_sec );
    usec   = double( tp.tv_usec )/1E6;
    return sec + usec;
}

#ifdef _OPENACC

void testacc(){
        
    double * x,               
           * y;

    const int N = 10;

    x = (double*) malloc( N*sizeof(double) );
    y = (double*) malloc( N*sizeof(double) );
        
    int i;
    #pragma acc parallel loop        
    for( i=0; i<N; i++ )                             
        x[i] = i;                     

    #pragma acc parallel loop
    for( i=0; i<N; i++ )
        y[i] = x[i]*x[i];
}
                                                     
void getColumn_acc_gpu_1d( double * A,  double *a, int rows, int colid){

#pragma acc kernels  
{
    #pragma acc loop independent
    for (int i = 0; i < rows; i++){
        a[i] = A[i + rows*colid];
    }
}

}


void updatev_acc_gpu_1d( double * vnew, double * vold, int rows, int colid, int zid, int cols, int ro_stepsp){
#pragma acc kernels  
{
    #pragma acc loop independent
    for (int i = 0; i < rows; i++){
        vnew [i + colid*rows   ] = vold [i + colid*rows  ] ;
    }
}
}


void cgsro_gpu( double * Q_1d,  double * v_1d, int ro_steps, int m, int n, double ** timer ){

    double timer_tmp;
    for(int ii = 0; ii < 8; ii++){
        timer[ro_steps-1][ii] = 0.0;
    }

    double time_cgs = gclock();
    timer_tmp = gclock();

    int ro_stepsp = ro_steps+1;

    double * aj = (double*)malloc(sizeof(double)*m); 

    // additional tables used in division into 2 stages caluclation of new v_1d
    double * tab_denominator = (double*)malloc(sizeof(double)*n);
    double * tab_tmp1 = (double*)malloc(sizeof(double)*n);
    
    for (int jj = 0; jj < n; jj++){
        tab_denominator[jj] = 0.0;
    }
    
    for(int tt=0; tt < n; tt++)    
        tab_tmp1[tt] = 0.0;

    int j, k;
    int row, col;

    #pragma acc enter data copyin(v_1d[0:m*n])
    #pragma acc enter data copyin(aj[0:m])
    #pragma acc enter data copyin(tab_tmp1[0:n])
    #pragma acc enter data copyin(tab_denominator[0:n])

    timer[ro_steps-1][0] += gclock() - timer_tmp;

    #pragma acc data copy(Q_1d[0:m*n])
    for ( j = 0; j < n; j++){

        //if ( j % 100 == 0)
        //    printf("CGS: column=%5d (%3.0f)\n", j, 100.0*(double)(j)/n );

        timer_tmp = gclock();
        
        getColumn_acc_gpu_1d( v_1d, aj, m, j);
        
        timer[ro_steps-1][1] += gclock() - timer_tmp;

        timer_tmp = gclock();

        timer[ro_steps-1][2] += gclock() - timer_tmp;
    

        double sqrttmp = 0.0;
        for ( k = 0; k < ro_steps; k++){
        
            timer_tmp = gclock();
            
            if (k==0)     
                updatev_acc_gpu_1d( Q_1d, v_1d, m, j, k, n, ro_stepsp );
            else
                updatev_acc_gpu_1d( v_1d, Q_1d, m, j, k, n, ro_stepsp );

            #pragma acc kernels
            {
              #pragma acc loop independent 
              for(col = 0; col < j; col++)          
                  tab_tmp1[col] = 0.0;
            }
            
            timer[ro_steps-1][3] += gclock() - timer_tmp;
            timer_tmp = gclock();
                
            #pragma acc kernels
            {
                #pragma acc loop independent         
                for ( int i = 0; i <= j-1; i++){

                    double tmp1 = 0.0;

                    #pragma acc loop reduction(+:tmp1)            
                    for ( int rowi = 0; rowi < m; rowi++){
                        tmp1 += (Q_1d[rowi+i*m]) * ( v_1d[rowi +  j*m ] ) ;            
                    }
                    tab_tmp1[i] = tmp1*tab_denominator[i];        
                }
            }
            
            // First stage [loop parallelizable] :
            #pragma acc kernels
            {
                #pragma acc loop independent device_type(nvidia) //gang worker (32)        
                for ( int i = 0; i <= j-1; i++){
                    #pragma acc loop independent device_type(nvidia) //vector(32)
                    for ( int rowi = 0; rowi < m; rowi++)            
                        v_1d[rowi + i*m  ] = tab_tmp1[i]* (Q_1d[rowi + i*m] *tab_denominator[i]) ;        
                }
            }// loop i < j-1
            
            // Second stage [loop reduction]
            #pragma acc kernels
            {
                #pragma acc loop independent  device_type(nvidia) //gang worker (256)
                for ( int rowi = 0; rowi < m; rowi++){
                    double tmpx = 0.0;
                    
                    #pragma acc loop reduction(+:tmpx)
                    for ( int i = 0; i <= j-1; i++){
                        tmpx += v_1d[rowi + i*m ];
                    }
                    Q_1d[rowi + j*m  ] -= tmpx; 
                }
            }

            timer[ro_steps-1][4] += gclock() - timer_tmp;
            timer_tmp = gclock();
            
            double tmp = 0.0;
            #pragma acc kernels
            {
                #pragma acc loop reduction(+:tmp)
                for ( int row = 0; row < m; row++)
                    tmp += Q_1d[ row + j*m ] * Q_1d[ row + j*m ] ;
            }
        
            sqrttmp = sqrt(tmp);
            
            timer[ro_steps-1][5] += gclock() - timer_tmp;

        }// end re-orthogonalization
        k--;
           
        timer_tmp = gclock();
        
        #pragma acc kernels
        {
        tab_denominator[j] = 1.0/sqrttmp; 
        }

        if ((j+1==n)){
            
            // update of Q for last column: 
            #pragma acc kernels
            {
                double denominator;
                #pragma acc loop independent
                for ( int jj = 0; jj < n; jj++){
                    denominator = tab_denominator[jj];
                    #pragma acc loop independent
                    for ( row = 0; row < m; row++){
                        Q_1d[row+jj*m ]  = Q_1d[row+jj*m] * denominator;
                    }
                }
            }
            
            //printf("j = %d, n = %d |END OF Q-updated|\n", j, n);
        }
        timer[ro_steps-1][6] += gclock() - timer_tmp;


    } // end loop over columns
    
    time_cgs = gclock() - time_cgs;


    double time_loop = 0.0;
    for(int ii = 0; ii < 7; ii++){
        time_loop += timer[ro_steps-1][ii];
    }
    timer[ro_steps-1][8] = time_loop;

    //need to be added to initialization time:
    double time_Q_HtoD_DtoH = time_cgs - time_loop;

    timer[ro_steps-1][0] = timer[ro_steps-1][0] + time_Q_HtoD_DtoH;
    timer[ro_steps-1][8] = timer[ro_steps-1][8] + time_Q_HtoD_DtoH;

    printf("[CGS-RO GPU] PHASE              sec. [ %% ] \n"  );
    printf("[CGS-RO GPU] 1. init         = %1.3f [%3.1f ] \n", timer[ro_steps-1][0], 100.0*timer[ro_steps-1][0] / time_cgs );
    printf("[CGS-RO GPU] 2. aj           = %1.3f [%3.1f ] \n", timer[ro_steps-1][1], 100.0*timer[ro_steps-1][1] / time_cgs );
    printf("[CGS-RO GPU] 3. vj           = %1.3f [%3.1f ] \n", timer[ro_steps-1][2], 100.0*timer[ro_steps-1][2] / time_cgs);
    printf("[CGS-RO GPU] 4. re-ortho     = %1.3f [%3.1f ] \n", timer[ro_steps-1][3]+timer[ro_steps-1][4]+timer[ro_steps-1][5], 100.0*(timer[ro_steps-1][3]+timer[ro_steps-1][4]+timer[ro_steps-1][5]) / time_cgs);
    printf("[CGS-RO GPU]  re-ortho(1)     = %1.2f [%3.1f ] \n", timer[ro_steps-1][3], 100.0*timer[ro_steps-1][3] / time_cgs);
    printf("[CGS-RO GPU]  re-ortho(2)     = %1.2f [%3.1f ] \n", timer[ro_steps-1][4], 100.0*timer[ro_steps-1][4] / time_cgs);
    printf("[CGS-RO GPU]  re-ortho(3)     = %1.2f [%3.1f ] \n", timer[ro_steps-1][5], 100.0*timer[ro_steps-1][5] / time_cgs);
    printf("[CGS-RO GPU] 5. Q            = %1.3f [%3.1f ] \n", timer[ro_steps-1][6], 100.0*timer[ro_steps-1][6] / time_cgs);
    printf("[CGS-RO GPU] 1-5 CGS-RO      = %1.3f [%3.1f ] \n", timer[ro_steps-1][8], 100.0*timer[ro_steps-1][8] / time_cgs);
    printf("|-----------------------------------\n");
    printf("[CGS-RO GPU] TRANSFERS       = %3.4f [%3.1f ] \n", timer[ro_steps-1][0], 100.0*timer[ro_steps-1][0] / time_cgs);
    printf("[CGS-RO GPU] COMPUTATIONS    = %3.4f [%3.1f ] \n", timer[ro_steps-1][8]-timer[ro_steps-1][0], 100.0*(timer[ro_steps-1][8]-timer[ro_steps-1][0]) / time_cgs);
    printf("[CGS-RO GPU] 1-5 CGS-RO      = %3.4f [%3.1f ] \n", timer[ro_steps-1][8], 100.0*timer[ro_steps-1][8] / time_cgs);
    printf("|-----------------------------------\n");

    //return tx;

}


#endif





