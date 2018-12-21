# matrix-mult
Matrix multiplication samples.

## Plain C

The c/ directory contains code that do matrix multiplication in plain
sequential C.

# POSIX threads

The pthread/ directory contains code that does matrix multiplication with
pthreads.

## OpenMP

The openmp/ directory contains code that does matrix multiplication in C with
OpenMP.

For optimized results, use affinity with OMP_PROC_BIND/OMP_PLACES and pass
number of cores for thread parameter.
Example for a quadcore hyperthreaded CPU:
OMP_PROC_BIND=spread OMP_PLACES=cores ./matmult-omp -t 4

It is also possible to specify explicitely the affinity with GOMP_CPU_AFFINITY
environment.
Example for a quadcore hyperthreaded CPU:
GOMP_CPU_AFFINITY=0,1,2,3 ./matmult-omp -t 4

Please note there is no single OpenMP configuration that works _best_ on every
CPU. It is recommended to study the architecture of CPUs with hwloc-ls and benchit.
In general for matrix calculation, best results are achieved by using no more
than number of core threads and by binding one thread on a different core (and
using only the first logical ones for hyperthreaded CPU).

## Message Passing Interface

The mpi/ directory contains code that does matrix multiplication in C with
MPI. There are classic MPI version and hybrid MPI/OpenMP version.

## OpenCL

The opencl/ directory contains code that does matrix multiplication in C with
OpenCL to offload calculation.

It contains three different OpenCL kernels. Some may perform better depending on
OpenCL ICD.

## License

All codes are under BSD-3 license.

