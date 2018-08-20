ParallelCGS: Parallel classical Gram-Schmidt algorithm with reorthogonalization (CGS-RO)
=============================

![Version](https://img.shields.io/badge/version-1.0-green.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](http://opensource.org/licenses/MIT)

Overview
--------

This project provides C++ codes for classical Gram-Schmidt algorithm with reorthogonalization (CGS-RO) [1] in which computations are performed sequentially (CPU) or in parallel with OpenACC directives (CPU, GPU).


Details
-------

In a file `main.cpp` the execution setup is settled and function `run_cgsro(...)` is called. As default a reference sequential CPU implementation (`cgsro_sequential(...)`) is performed and dependingly on the target it is compared with one of the selected parallel implementation :
 - `cgsro_multicore(...)` on a CPU, or 
 - `cgsro_gpu(...)` on a GPU. 

The pseudo code of a CGS-RO algorithm is presented in listing below. One can distinguish two main loops: 
- lines 1-15 over columns of the matrix `A`
- inner 4-12 in which re-orthogonalization is performed

```
1. for(j=0;j<n;j++){
2.    aj = A(:,j);
3.    v(:,j,0) = aj;
4.    for(k=0;k<ro_steps;k++){
5.        v(:,j:,k+1) = v(:,j,k)
6.        for(i=0;i<=j;i++){
7.            tmp1 = sum (Q(:,j) .* v(:,j,k)) // reduction 
8.            v(:,j,k+1) = v(:,j,k+1)-tmp1*Q(:,i)
9.        }
10.       tmp = sum (v(:,j,k+1) .* v(:,j,k+1)) // reduction
11.       sqrttmp = sqrt(tmp)
12.    }
13.    k--;
14.    Q(:,j) = v(:,j,k)/sqrttmp;
15. }
```

There are several steps that can be parallelized: reduction (lines: 7, 10), copy (lines: 2,3,5), axpy (line: 8) and scal (lines: 14). Thus, OpenACC was used used to parallelize computations on a CPU  (`cgsro_multicore.cpp`). 

However, in GPU-accelerated implementations three key modifications were applied in order to significantly save memory  and achieve better performance (`cgsro_gpu.cpp`):
- originally, the size of matrix `v` is `m x n x ro_steps`, where `m` and `n` are the number of rows and columns of matrix `A` and `ro_steps` is the number of reorthogonalization steps. In the approach proposed here the size of matrix `v` was significantly reduced since matrix `v` requires only `m x n x 2`. In this case, the temporary results of `Q` are stored in `v` through the entire Gram-Schmidt orthogonalization process and final update of `Q` is done for the last
column after reorthogonalization is finished . To make it possible computations are rearranged an additional table of relatively smaller size is needed (`tab_denominator`).
-  to achieve a significantly better performance on a GPU the original loop (line 8), in which ’a new’ `v` is calculated, was divided into two stages (see `cgsro_gpu.cpp`). To make it possible an additional table of relatively smaller size is needed (`tab_tmp1`). Moreover, in the second stage the order of performing computations was changed (outer loop over rows, inner over columns), which allowed achieving better results on a GPU.

Notation: the input matrix is stored in two dimensional array, however, it is copied to one dimensional array (`A_1d`) which was more convenient  for accessing data in GPU implementation. For the same reason other tables are also one dimensional (i.e. `Q_1d`, `v_1d`).

Author
------

Author: Adam Dziekonski, Dr.Eng.
Date of publication: August 2018



Software requirements
----------------------------------
PGI Community Edition 17.10




Compilation
-----------
Compilation and run procedures for Linux are gathered in file `compile.sh`. 

Few suggestion about PGI flags:
- to perform computations on a CPU sequentially or in parallel one has to set target: `-ta=host` or `-ta=multicore`, respectively,
- to execute code on a GPU a flag `-ta=tesla:...` with a proper parameter which selects a graphics accelerator is needed,
- do not forget to use a flag `-Mlarge_arrays` in order to allocate large arrays in global memory on a GPU,
- to show information how OpenACC manages to select execution setup, which loop is parallelizable use `-Minfo=accel` flag. 

Example
-------
If one want to run CGS-RO for 100000 x 100 matrix and set three re-orthogonalizations, then:
- in order to compare CPU (sequential) with CPU (multicore) implementations use the following: `./cgsro_multicore 100000 100 3 1`

- in order to compare CPU (sequential) with GPU implementations use the following: `./cgsro_tesla 100000 100 3 2`


Below the shortened output of CGS-RO in which a sequential and GPU-accelerated implementations are compared for matrix with 100000 rows and 100 columns is presented for execution: `./cgsro_tesla 100000 100 1 2`

```
ParallelCGS: classical Gram-Schmidt with re-orthogonalization:
CGS setup >>> m(rows) = 100000, n(cols) = 100, ro_steps = 1, target = 2
A [100000 x 100]
CGS-RO (reference: sequential on a CPU) :
[CGS-RO SEQUENTIAL] PHASE      sec. [ % ]
[CGS-RO SEQUENTIAL] CGS-RO   = 1.268 [100.0 ]
[CGS-RO][re-ortho #1] NormInf(I-Q^T*Q) = 2.753e-14

CGS-RO with OPENACC:
CGS-RO (TARGET=GPU):
[CGS-RO GPU] PHASE              sec. [ % ]
[CGS-RO GPU] 1-5 CGS-RO      = 0.170 [100.0 ]
[CGS-RO][re-ortho #1] NormInf(I-Q^T*Q) =  2.931e-14

Speedup [CGS-RO][# re-orthogonalizations =  1] = 7.47
```
Results obtained for a CPU: Intel Xeon E5-2680 v3 and  GPU: Tesla K40.


References
-----------
[1] Giraud, Luc, Julien Langou, and Miroslav Rozloznik. "The loss of orthogonality in the GramSchmidt orthogonalization process." Computers Mathematics with Applications 50.7 (2005): 1069-1075.


License
-------

ParallelCGS is an open-source C++ code licensed under the [MIT license](http://opensource.org/licenses/MIT).





