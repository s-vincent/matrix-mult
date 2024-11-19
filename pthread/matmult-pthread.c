/*
 * Copyright (c) 2014-2017, Sebastien Vincent
 *
 * Distributed under the terms of the BSD 3-clause License.
 * See the LICENSE file for details.
 */

/**
 * \file matmult-pthread.c
 * \brief Matrix multiplication in C/pthread.
 * \author Sebastien Vincent
 * \date 2014-2017
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

#include <pthread.h>

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
 * \brief Thread worker data.
 */
struct mat_mult_data
{
  /**
   * \brief Index of worker.
   */
  size_t idx;

  /**
   * \brief First matrix.
   */
  uint64_t* mat1;

  /**
   * \brief Second matrix.
   */
  uint64_t* mat2;

  /**
   * \brief Result matrix.
   */
  uint64_t* result;

  /**
   * \brief Row size.
   */
  size_t m;

  /**
   * \brief Colummn size.
   */
  size_t n;

  /**
   * \brief Colummn size.
   */
  size_t w;

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
 * \brief Thread worker to calculate matrix.
 * \param data data.
 * \return NULL;
 */
static void* mat_mult_work(void* data)
{
  struct mat_mult_data* d = (struct mat_mult_data*)data;
  uint64_t* mat1 = d->mat1;
  uint64_t* mat2 = d->mat2;
  uint64_t* result = d->result;
  size_t m = d->m;
  size_t n = d->n;
  size_t w = d->w;

  size_t step = m / d->threads;

  if(step == 0)
  {
    step = 1;

    /* exit threads that will not work */
    if(d->idx >= m)
    {
      return NULL;
    }
  }

  for(size_t i = d->idx * step ; i < (d->idx + 1) * step && i < m; i++)
  {
    for(size_t j = 0 ; j < n ; j++)
    {
      uint64_t tmp = 0;

      for(size_t k = 0 ; k < w ; k++)
      {
        tmp += mat1[i * w + k] * mat2[k * n + j];
      }

      result[i * m + j] = tmp;
    }
  }

  return NULL;
}

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
void mat_print(uint64_t* mat, size_t m, size_t n)
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
 * \param threads thread number.
 * \return 0 if success, -1 if matrixes cannot be multiplied.
 */
int mat_mult_pthread(uint64_t* mat1, uint64_t* mat2, uint64_t* result, size_t m,
    size_t n, size_t w, size_t threads)
{
  pthread_t ids[threads];
  struct mat_mult_data datas[threads];

  if(n != w)
  {
    return -1;
  }

  for(size_t i = 0 ; i < threads ; i++)
  {
    int ret = 0;

    datas[i].idx = i;
    datas[i].mat1 = mat1;
    datas[i].mat2 = mat2;
    datas[i].result = result;
    datas[i].m = m;
    datas[i].n = n;
    datas[i].w = w;
    datas[i].threads = threads;

    ret = pthread_create(&ids[i], NULL, mat_mult_work, &datas[i]);

    if(ret != 0)
    {
      errno = ret;
      perror("pthread_create error");
      return -1;
    }
  }

  for(size_t i = 0 ; i < threads ; i++)
  {
    pthread_join(ids[i], NULL);
  }

  return 0;
}

/**
 * \brief Print help.
 * \param program program name.
 */
void print_help(const char* program)
{
  fprintf(stdout, "Usage: %s [-m row size] [-t nb] "
      "[-p] [-h]\n\n"
      "  -h\t\tDisplay this help\n"
      "  -p\t\tPrint the input and output matrixes\n"
      "  -m row\tRow/column size (default 1024)\n"
      "  -t nb\t\tNumber of threads to use\n", program);
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
  uint64_t* mat1 = NULL;
  uint64_t* mat2 = NULL;
  uint64_t* mat3 = NULL;
  size_t m = DEFAULT_ROW_SIZE;
  size_t n = DEFAULT_COLUMN_SIZE;
  size_t w = DEFAULT_COLUMN_SIZE;
  int print_matrix = 0;
  size_t threads = 0;
  struct configuration config;
  size_t nb_elements = 0;
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
  n = config.m;
  w = config.m;
  print_matrix = config.print_matrix;
  threads = config.threads;

  nb_elements = m * n;

  mat1 = malloc(nb_elements * sizeof(uint64_t));
  mat2 = malloc(nb_elements * sizeof(uint64_t));
  mat3 = malloc(nb_elements * sizeof(uint64_t));

  if(!mat1 || !mat2 || !mat3)
  {
    perror("malloc");
    free(mat1);
    free(mat2);
    free(mat3);
    exit(EXIT_FAILURE);
  }

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
  if(mat_mult_pthread(mat1, mat2, mat3, m, n, w, threads) == -1)
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

