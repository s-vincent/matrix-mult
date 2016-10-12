/*
 * Copyright (c) 2014-2016, Sebastien Vincent
 *
 * Distributed under the terms of the BSD 3-clause License.
 * See the LICENSE file for details.
 */

/**
 * \file matmult-omp.c
 * \brief Matrix multiplication in C/OpenMP.
 * \author Sebastien Vincent
 * \date 2014-2016
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

#include <sys/time.h>

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
   * \brief Row size.
   */
  size_t m;

  /**
   * \brief Colummn size.
   */
  size_t n;

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
 * \brief Get time in microseconds.
 * \return time in microseconds.
 */
static double util_gettime_us(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000000 + t.tv_usec;
}

/**
 * \brief Initializes the matrixes.
 * \param mat1 first matrix.
 * \param mat2 second matrix.
 * \param m row size of the matrix.
 * \param n column size of the matrix.
 */
void mat_init(int* mat1, int* mat2, size_t m, size_t n)
{
  for(size_t i = 0 ; i < (m * n) ; i++)
  {
    mat1[i] = rand();
    mat2[i] = rand();
  }
}

/**
 * \brief Print the matrix content on stdout.
 * \param mat the matrix.
 * \param m row size of the matrix.
 * \param n column size of the matrix.
 */
void mat_print(int* mat, size_t m, size_t n)
{
  for(size_t i = 0 ; i < m ; i++)
  {
    for(size_t j = 0 ; j < n ; j++)
    {
      fprintf(stdout, "%d ", mat[i * m + j]);
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
 * \param threads thread number.
 * \return 0 if success, -1 if matrixes cannot be multiplied.
 */
int mat_mult_omp(int* mat1, int* mat2, int* result, size_t m, size_t n,
    size_t w, size_t threads)
{
  if(n != w)
  {
    return -1;
  }

  /* to set spread way, add to next line: proc_bind(spread) */
  #pragma omp parallel num_threads(threads)
  for(size_t i = 0 ; i < m ; i++)
  {
    #pragma omp for schedule(static)
    for(size_t j = 0 ; j < n ; j++)
    {
      int tmp = 0;

      for(size_t k = 0 ; k < w ; k++)
      {
        tmp += mat1[i * w + k] * mat2[k * n + j];
      }

      result[i * m + j] = tmp;
    }
  }

  return 0;
}

/**
 * \brief Print help.
 * \param program program name.
 */
void print_help(const char* program)
{
  fprintf(stdout, "Usage: %s [-m row size] [-n column size] [-t nb] "
      "[-p] [-h]\n\n"
      "  -h\t\tDisplay this help\n"
      "  -p\t\tPrint the input and output matrixes\n"
      "  -m row\tDefine row size (default 1024)\n"
      "  -n col\tDefine column size (default 1024)\n"
      "  -t nb\t\tDefines number of threads to use\n", program);
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
   * n: column size
   * t: number of threads to use
   */
  static const char* options = "hpm:n:t:";
  int opt = 0;
  int print_matrix = 0;
  long m = DEFAULT_ROW_SIZE;
  long n = DEFAULT_COLUMN_SIZE;
  int ret = 1;
  int threads = sysconf(_SC_NPROCESSORS_ONLN);

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
      case 'n':
        n = atol(optarg);
        if(n < 2)
        {
          fprintf(stderr, "Bad argument for '-n' %ld\n", n);
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
  configuration->n = n;
  configuration->threads = (size_t)threads;

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
  int* mat1 = NULL;
  int* mat2 = NULL;
  int* mat3 = NULL;
  size_t m = DEFAULT_ROW_SIZE;
  size_t n = DEFAULT_COLUMN_SIZE;
  size_t w = DEFAULT_COLUMN_SIZE;
  int print_matrix = 0;
  size_t threads = 0;
  struct configuration config;
  double start = 0;
  double end = 0;
  int ret = 0;

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
  n = config.n;
  w = config.n;
  print_matrix = config.print_matrix;
  threads = config.threads;

  mat1 = malloc((m * n) * sizeof(int));
  mat2 = malloc((m * n) * sizeof(int));
  mat3 = malloc((m * n) * sizeof(int));

  if(!mat1 || !mat2 || !mat3)
  {
    perror("malloc");
    free(mat1);
    free(mat2);
    free(mat3);
    exit(EXIT_FAILURE);
  }

  /* random initialization */
  srand(time(NULL));

  mat_init(mat1, mat2, m, n);

  if(print_matrix)
  {
    printf("Matrix 1:\n");
    mat_print(mat1, m, n);
    printf("Matrix 2:\n");
    mat_print(mat2, m, n);
  }

  fprintf(stdout, "Compute with %zu thread(s)\n", threads);

  start = util_gettime_us();
  if(mat_mult_omp(mat1, mat2, mat3, m, n, w, threads) == -1)
  {
    fprintf(stderr, "Matrixes cannot be multiplied\n");
    ret = EXIT_FAILURE;
  }
  else
  {
    end = util_gettime_us();

    fprintf(stdout, "Multiplication success: %f ms\n", (end - start) / 1000);

    if(print_matrix)
    {
      mat_print(mat3, m, n);
    }
    ret = EXIT_SUCCESS;
  }

  /* free resources */
  free(mat1);
  free(mat2);
  free(mat3);

  return ret;
}

