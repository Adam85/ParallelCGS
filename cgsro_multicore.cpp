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
 Content of cgsro_multicore.cpp : this function incudes a multicore OpenACC implementation for a CPU  
 Author: Adam Dziekonski
 Generated August 2018
***************************************************************************************/

#include "helpers.h"
#include "cgsro_multicore.h"

double tclock(){
    struct timeval tp;
    double sec,
           usec;

    gettimeofday( &tp, NULL );
    sec    = double( tp.tv_sec );
    usec   = double( tp.tv_usec )/1E6;
    return sec + usec;
}

void getColumn_acc_1d( double * A,  double *a, int rows, int colid){
    #pragma acc parallel loop
    for (int i = 0; i < rows; i++){
        a[i] = A[i + rows*colid];
    }
}

void setColumn_acc_1d( double * A,  double *a, int rows, int colid, int zid, int cols){
    #pragma acc parallel loop
    for (int i = 0; i < rows; i++){
        A[i+   colid*rows +  zid*(rows*cols)] = a[i] ;
    }
}

void updatev_acc_1d( double * A, int rows, int colid, int zid, int cols){
    #pragma acc parallel loop
    for (int i = 0; i < rows; i++){
        A[i + colid*rows  + (zid+1)*(rows*cols) ] = A[i + colid*rows + zid * (rows*cols) ] ;
    }
}

void cgsro_multicore( double * A_1d,  double * Q_1d,  int ro_steps, int m, int n, double ** timer){

    double timer_tmp;
    for(int ii = 0; ii < 8; ii++){
        timer[ro_steps-1][ii] = 0.0;
    }

    double time_cgs = tclock();
    timer_tmp = tclock();

    double * v_1d = (double*)malloc(sizeof(double)*m*n*(ro_steps+1));
    double * aj = (double*)malloc(sizeof(double)*m); 

    int j, i, k;
    int row;

    timer[ro_steps-1][0] += tclock() - timer_tmp;

    for ( j = 0; j < n; j++){

        //if ( j % 100 == 0)
        //    printf("CGS: column=%5d (%3.0f)\n", j, 100.0*(double)(j)/n );

        timer_tmp = tclock();
        getColumn_acc_1d( A_1d, aj, m, j);
        timer[ro_steps-1][1] += tclock() - timer_tmp;

        timer_tmp = tclock();
        setColumn_acc_1d( v_1d, aj, m, j, 0, n );
        timer[ro_steps-1][2] += tclock() - timer_tmp;
    

        double sqrttmp = 0.0;
        for ( k = 0; k < ro_steps; k++){
        
            timer_tmp = tclock();
            updatev_acc_1d( v_1d, m, j, k, n);
            timer[ro_steps-1][3] += tclock() - timer_tmp;
            

            timer_tmp = tclock();
            for ( i = 0; i <= j-1; i++){
             
                double tmp1  = 0.0;
                {
                    #pragma acc parallel loop reduction(+:tmp1) 
                    for ( int rowi = 0; rowi < m; rowi++){
                        tmp1 += Q_1d[rowi+i*m] * v_1d[rowi +  j*m + k*m*n];
                    }
                }

                {
                    #pragma acc parallel loop
                    for ( int rowi = 0; rowi < m; rowi++)
                        v_1d[rowi + j*m + (k+1)*m*n ] = v_1d[rowi + j*m + (k+1)*m*n ] - tmp1*Q_1d[rowi + i*m];

                }


            }
            timer[ro_steps-1][4] += tclock() - timer_tmp;
             
 
            timer_tmp = tclock();
            double tmp = 0.0;
            
            for ( row = 0; row < m; row++)
                tmp += v_1d[row + j*m + (k+1)*m*n ] * v_1d[row + j*m + (k+1)*m*n ] ;
            
            sqrttmp = sqrt(tmp);
        
            timer[ro_steps-1][5] += tclock() - timer_tmp;
            


        }// end re-orthogonalization
        k--;
           
        timer_tmp = tclock();
        #pragma acc parallel loop 
        for ( row = 0; row < m; row++){
            Q_1d[row+j*m] = v_1d[row + j*m + (k+1)*m*n]/sqrttmp;
        }
        timer[ro_steps-1][6] += tclock() - timer_tmp;


    } // end loop over columns
    
    time_cgs = tclock() - time_cgs;

    double time_loop = 0.0;
    for(int ii = 0; ii < 7; ii++){
        time_loop += timer[ro_steps-1][ii];
    }
    timer[ro_steps-1][8] = time_loop;

    printf("[CGS-RO MULTICORE] PHASE               sec. [ %% ] \n"  );
    printf("[CGS-RO MULTICORE] 1. init          = %1.3f [%3.1f ] \n", timer[ro_steps-1][0], 100.0*timer[ro_steps-1][0] / time_cgs );
    printf("[CGS-RO MULTICORE] 2. aj            = %1.3f [%3.1f ] \n", timer[ro_steps-1][1], 100.0*timer[ro_steps-1][1] / time_cgs );
    printf("[CGS-RO MULTICORE] 3. vj            = %1.3f [%3.1f ] \n", timer[ro_steps-1][2], 100.0*timer[ro_steps-1][2] / time_cgs);
    printf("[CGS-RO MULTICORE] 4. re-ortho      = %1.3f [%3.1f ] \n", timer[ro_steps-1][3]+timer[ro_steps-1][4]+timer[ro_steps-1][5], 100.0*(timer[ro_steps-1][3]+timer[ro_steps-1][4]+timer[ro_steps-1][5]) / time_cgs);
    printf("[CGS-RO MULTICORE]  re-ortho(1)      = %1.2f [%3.1f ] \n", timer[ro_steps-1][3], 100.0*timer[ro_steps-1][3] / time_cgs);
    printf("[CGS-RO MULTICORE]  re-ortho(2)      = %1.2f [%3.1f ] \n", timer[ro_steps-1][4], 100.0*timer[ro_steps-1][4] / time_cgs);
    printf("[CGS-RO MULTICORE]  re-ortho(3)      = %1.2f [%3.1f ] \n", timer[ro_steps-1][5], 100.0*timer[ro_steps-1][5] / time_cgs);
    printf("[CGS-RO MULTICORE] 5. Q             = %1.3f [%3.1f ] \n", timer[ro_steps-1][6], 100.0*timer[ro_steps-1][6] / time_cgs);
    printf("[CGS-RO MULTICORE] 1-5 CGS-RO       = %1.3f [%3.1f ] \n", timer[ro_steps-1][8], 100.0*timer[ro_steps-1][8] / time_cgs);

}




