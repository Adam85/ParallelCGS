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
 Content of cgsro_sequential.cpp : this function incudes a reference sequential implementation on a CPU  
 Author: Adam Dziekonski
 Generated August 2018
***************************************************************************************/



#include "helpers.h"
#include "cgsro_sequential.h"

void getColumn_1d( double * A,  double *a, int rows, int colid){
    for (int i = 0; i < rows; i++){
        a[i] = A[i + colid*rows];
    }
}

void setColumn_1d( double * A,  double *a, int rows, int colid, int zid, int cols){
    for (int i = 0; i < rows; i++){
        A[i + colid*rows + zid*rows*cols] = a[i] ;
    }
}

void updatev_1d( double * A, int rows, int colid, int zid, int cols){
    for (int i = 0; i < rows; i++){
        A[i + colid*rows + (zid+1)*rows*cols] = A[i + colid*rows + zid*rows*cols ] ;
    }
}

void  cgsro_sequential( double * A_1d,  double * Q_1d, int steps, int m, int n, double ** timer){

    double timer_tmp;
    for(int ii = 0; ii < 9; ii++){
        timer[steps-1][ii] = 0.0;
    }

    double time_cgs = mclock();
    timer_tmp = mclock();

    double * aj = (double*)malloc(sizeof(double)*m);
    double * v_1d = (double*)malloc(sizeof(double)*m*n*(steps+1));
    double * r_1d = (double*)malloc(sizeof(double)*m*n*(steps  ));

    int j, i, k;
    int row;//, col;


    timer[steps-1][0] += mclock() - timer_tmp;

    double sqrttmp = 0.0;
    for ( j = 0; j < n; j++){

        //if ( j % 100 == 0)
        //    printf("CGS: column=%5d (%3.0f)\n", j, 100.0*(double)(j)/n );

        timer_tmp = mclock();
        getColumn_1d( A_1d, aj, m, j);

        timer[steps-1][1] += mclock() - timer_tmp;

        timer_tmp = mclock();
        setColumn_1d( v_1d, aj, m, j, 0, n);
        timer[steps-1][2] += mclock() - timer_tmp;

        // start re-orthogonalization
        for ( k = 0; k < steps; k++){
        
            timer_tmp = mclock();
            updatev_1d( v_1d, m, j, k, n);
            timer[steps-1][3] += mclock() - timer_tmp;

            timer_tmp = mclock();
            for ( i = 0; i <= j-1; i++){
      
                double tmp1 = 0.0;
                for ( row = 0; row < m; row++){
                    tmp1 += Q_1d[row + i*m] * v_1d[row + j*m + k*m*n];
                }

                for ( row = 0; row < m; row++)
                    v_1d[row + j*m+ (k+1)*m*n] = v_1d[row + j*m + (k+1)*m*n] - tmp1*Q_1d[row + i*m];
                
            }
            timer[steps-1][4] += mclock() - timer_tmp;
             
            
            timer_tmp = mclock();
            double tmp = 0.0;
            for ( row = 0; row < m; row++)
                tmp += v_1d[row+j*m+(k+1)*m*n] * v_1d[row+j*m+(k+1)*m*n] ;
            
            sqrttmp = sqrt ( tmp );
            timer[steps-1][5] += mclock() - timer_tmp;
            
        }// end re-orthogonalization
        k--;
            
        timer_tmp = mclock();
        for ( row = 0; row < m; row++){
            Q_1d[row + j*m] = v_1d[row+j*m+(k+1)*m*n]/sqrttmp;
        }
        timer[steps-1][6] += mclock() - timer_tmp;


    } // end loop over columns
    
    time_cgs = mclock() - time_cgs;

    double time_loop = 0.0;
    for(int ii = 0; ii < 7; ii++){
        time_loop += timer[steps-1][ii];
    }
    timer[steps-1][8] = time_loop;

    printf("[CGS-RO SEQUENTIAL] PHASE           sec. [ %% ] \n"  );
    printf("[CGS-RO SEQUENTIAL] 1. init      = %1.3f [%3.1f ] \n", timer[steps-1][0], 100.0*timer[steps-1][0] / time_cgs );
    printf("[CGS-RO SEQUENTIAL] 2. aj        = %1.3f [%3.1f ] \n", timer[steps-1][1], 100.0*timer[steps-1][1] / time_cgs );
    printf("[CGS-RO SEQUENTIAL] 3. vj        = %1.3f [%3.1f ] \n", timer[steps-1][2], 100.0*timer[steps-1][2] / time_cgs);
    printf("[CGS-RO SEQUENTIAL] 4. re-ortho  = %1.3f [%3.1f ] \n", timer[steps-1][3]+timer[steps-1][4]+timer[steps-1][5], 100.0*(timer[steps-1][3]+timer[steps-1][4]+timer[steps-1][5]) / time_cgs);
    printf("[CGS-RO SEQUENTIAL]  re-ortho(1)  = %1.2f [%3.1f ] \n", timer[steps-1][3], 100.0*timer[steps-1][3] / time_cgs);
    printf("[CGS-RO SEQUENTIAL]  re-ortho(2)  = %1.2f [%3.1f ] \n", timer[steps-1][4], 100.0*timer[steps-1][4] / time_cgs);
    printf("[CGS-RO SEQUENTIAL]  re-ortho(3)  = %1.2f [%3.1f ] \n", timer[steps-1][5], 100.0*timer[steps-1][5] / time_cgs);
    printf("[CGS-RO SEQUENTIAL] 5. Q         = %1.3f [%3.1f ] \n", timer[steps-1][6], 100.0*timer[steps-1][6] / time_cgs);
    printf("[CGS-RO SEQUENTIAL] 1-5 CGS-RO   = %1.3f [%3.1f ] \n", timer[steps-1][8], 100.0*timer[steps-1][8] / time_cgs);


}


