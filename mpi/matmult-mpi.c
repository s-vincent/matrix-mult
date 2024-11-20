/*
 * Copyright (c) 2014-2017, Sebastien Vincent
 *
 * Distributed under the terms of the BSD 3-clause License.
 * See the LICENSE file for details.
 */

/**
 * \file matmult-mpi.c
 * \brief Matrix multiplication in C/MPI.
 * \author Sebastien Vincent
 * \date 2018
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <time.h>

#include <sys/time.h>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * \brief Default row size.
 */
static const size_t DEFAULT_ROW_SIZE = 1024;

/**
 * \brief Default column size.
 */
static const size_t DEFAULT_COLUMN_SIZE = 1024;

/**
 * \struct configuration
 * \brief Configuration.
 */
struct configuration
{
  /**
   * \brief Row/column size.
   */
  size_t m;

  /**
   * \brief Print input and output matrixes.
   */
  int print_matrix;

  /**
   * \brief Number of threads.
   */
  size_t threads;
};

/**
 * \brief Initializes the matrixes.
 * \param mat1 first matrix.
 * \param mat2 second matrix.
 * \param m row size of the matrix.
 * \param n column size of the matrix.
 */
void mat_init(uint64_t* mat1, uint64_t* mat2, size_t m, size_t n)
{
  for(size_t i = 0 ; i < (m * n) ; i++)
  {
    mat1[i] = i;
    mat2[i] = i;
  }
}

/**
 * \brief Print the matrix content on stdout.
 * \param mat the matrix.
 * \param m row size of the matrix.
 * \param n column size of the matrix.
 */
void mat_print(uint64_t * mat, size_t m, size_t n)
{
  for(size_t i = 0 ; i < m ; i++)
  {
    for(size_t j = 0 ; j < n ; j++)
    {
      fprintf(stdout, "%lu ", mat[i * m + j]);
    }
    fprintf(stdout, "\n");
  }
}

/**
 * \brief Performs multiplication of matrixes.
 * \param mat1 first matrix.
 * \param mat2 second matrix.
 * \param result result matrix.
 * \param m row size of first matrix.
 * \param n column size of first matrix.
 * \param w row size of second matrix.
 * \param rank MPI rank.
 * \param world_size Total number of MPI nodes.
 * \param threads number of threads to use (OpenMP only).
 * \return 0 if success, -1 if matrixes cannot be multiplied.
 */
int mat_mult_mpi(uint64_t* mat1, uint64_t* mat2, uint64_t* result, size_t m,
    size_t n, size_t w, size_t rank, size_t world_size, size_t threads)
{
  int nb_elements = m * n;
  int nb_subelements = nb_elements / world_size;
  uint64_t* res = malloc(sizeof(uint64_t) * nb_subelements);

  (void)rank;
  (void)threads;

  if(n != w || !res)
  {
    return -1;
  }

  /* transmit row to each process */
  MPI_Scatter(mat1, nb_subelements, MPI_UINT64_T, mat1, nb_subelements,
      MPI_UINT64_T, 0 /* rank root */, MPI_COMM_WORLD);

  /* broadcast second matrix to other nodes */
  MPI_Bcast(mat2, nb_elements, MPI_UINT64_T, 0, MPI_COMM_WORLD);

	/* matrix multiply */

#if _OPENMP
  /* to set spread way, add to next line: proc_bind(spread) */
  #pragma omp parallel num_threads(threads)
#endif
  for(size_t i = 0 ; i < (m / world_size) ; i++)
  {
#if _OPENMP
    #pragma omp for schedule(static)
#endif
    for(size_t j = 0 ; j < n ; j++)
    {
      uint64_t tmp = 0;

      for(size_t k = 0 ; k < w ; k++)
      {
        tmp += mat1[i * w + k] * mat2[k * n + j];
      }

      res[i * m + j] = tmp;
    }
  }

  MPI_Gather(res, nb_subelements, MPI_UINT64_T, result, nb_subelements,
      MPI_UINT64_T, 0, MPI_COMM_WORLD);

  free(res);
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

/**
 * \brief Print help.
 * \param program program name.
 */
void print_help(const char* program)
{
  fprintf(stdout, "Usage: %s [-m row size] "
#ifdef _OPENMP
      "[-t thread_number]"
#endif
      "[-p] [-h]\n\n"
      "  -h\t\tDisplay this help\n"
#ifdef _OPENMP
      "  -t nb\t\tDefines number of threads to use\n"
#endif
      "  -p\t\tPrint the input and output matrixes\n"
      "  -m row\tRow/column size (default 1024)\n",
      program);
}

/**
 * \brief Parse command line.
 * \param argc number of arguments.
 * \param argv array of arguments.
 * \param configuration configuration parameters.
 * \return 0 to exit with success, -1 to exit with error, otherwise continue.
 */
int parse_cmdline(int argc, char** argv,
    struct configuration* configuration)
{
  /*
   * h: print help and exit
   * p: print input and output matrixes
   * m: row size
   * t: number of threads to use
   */
  static const char* options = "hpm:t:";
  int opt = 0;
  int print_matrix = 0;
  long m = DEFAULT_ROW_SIZE;
  int threads = sysconf(_SC_NPROCESSORS_ONLN);
  int ret = 1;

  assert(configuration);

  while((opt = getopt(argc, argv, options)) != -1)
  {
    switch(opt)
    {
      case 'h':
        /* help */
        print_help(argv[0]);
        return 0;
        break;
      case 'p':
        print_matrix = 1;
        break;
      case 'm':
        m = atol(optarg);
        if(m < 2)
        {
          fprintf(stderr, "Bad argument for '-m' %ld\n", m);
          ret = -1;
        }
        break;
      case 't':
        threads = atol(optarg);
        if(threads <= 0)
        {
          fprintf(stderr, "Bad argument for '-t': %s\n", optarg);
          ret = EXIT_FAILURE;
        }
        break;
      default:
        fprintf(stderr, "Bad option (%c)\n", optopt);
        ret = -1;
        break;
    }
  }

  configuration->print_matrix = print_matrix;
  configuration->m = m;
#ifdef _OPENMP
  configuration->threads = threads;
#else
  configuration->threads = 1;
#endif

  return ret;
}

/**
 * \brief Entry point of the program.
 * \param argc number of arguments.
 * \param argv array of arguments.
 * \return EXIT_SUCCESS or EXIT_FAILURE.
 */
int main(int argc, char** argv)
{
  uint64_t* mat1 = NULL;
  uint64_t* mat2 = NULL;
  uint64_t* mat3 = NULL;
  size_t m = DEFAULT_ROW_SIZE;
  size_t n = DEFAULT_COLUMN_SIZE;
  size_t w = DEFAULT_COLUMN_SIZE;
  int print_matrix = 0;
  struct configuration config;
  double start = 0;
  double end = 0;
  int ret = 0;
  int world_size = 0;
  int world_rank = 0;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len = 0;
  size_t nb_elements = 0;

  ret = parse_cmdline(argc, argv, &config);

  if(ret == 0)
  {
    exit(EXIT_SUCCESS);
  }
  else if(ret == -1)
  {
    exit(EXIT_FAILURE);
  }

  m = config.m;
  n = config.m;
  w = config.m;
  print_matrix = config.print_matrix;

  nb_elements = m * n;

  /* MPI initialization */
#if _OPENMP
  int required = MPI_THREAD_SERIALIZED;
  int provided = 0;

  if(MPI_Init_thread(NULL, NULL, required, &provided) != MPI_SUCCESS)
  {
    fprintf(stderr, "Failed to initialize MPI.\n");
    exit(EXIT_FAILURE);
  }

  if(provided < required)
  {
    fprintf(stderr, "Failed to configure MPI thread.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

#else
  if(MPI_Init(NULL, NULL) != MPI_SUCCESS)
  {
    fprintf(stderr, "Failed to initialize MPI.\n");
    exit(EXIT_FAILURE);
  }
#endif

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Get_processor_name(processor_name, &name_len);

  if(m % world_size)
  {
    if(world_rank == 0)
    {
      fprintf(stderr,
          "Matrix size (%zu) not divisible by number of processor (%d)\n",
          nb_elements,
          world_size);
    }

    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  fprintf(stdout, "MPI from processor %s, rank %d out of %d\n",
      processor_name, world_rank, world_size);

  mat1 = malloc(nb_elements * sizeof(uint64_t));
  mat2 = malloc(nb_elements * sizeof(uint64_t));
  mat3 = malloc(nb_elements * sizeof(uint64_t));

  if(!mat1 || !mat2 || !mat3)
  {
    perror("malloc");
    free(mat1);
    free(mat2);
    free(mat3);

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if(world_rank == 0)
  {
    mat_init(mat1, mat2, m, n);

    if(print_matrix)
    {
      fprintf(stdout, "Matrix 1:\n");
      mat_print(mat1, m, n);
      fprintf(stdout, "Matrix 2:\n");
      mat_print(mat2, m, n);
    }

    fprintf(stdout, "Compute with %zu MPI node(s) with %zu thread(s) \n",
        (size_t)world_size, config.threads);
  }

  start = MPI_Wtime();
  if(mat_mult_mpi(mat1, mat2, mat3, m, n, w, world_rank, world_size,
        config.threads) == -1)
  {
    fprintf(stderr, "Matrixes cannot be multiplied\n");
    ret = EXIT_FAILURE;
  }
  else
  {
    end = MPI_Wtime();

    if(world_rank == 0)
    {
      fprintf(stdout, "Multiplication success: %f ms\n", (end - start) * 1000);

      if(print_matrix)
      {
        mat_print(mat3, m, n);
      }
    }
    ret = EXIT_SUCCESS;
  }

  /* free resources */
  free(mat1);
  free(mat2);
  free(mat3);

  MPI_Finalize();

  return ret;
}

