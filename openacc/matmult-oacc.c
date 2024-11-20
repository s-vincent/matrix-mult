/*
 * Copyright (c) 2023, Sebastien Vincent
 *
 * Distributed under the terms of the BSD 3-clause License.
 * See the LICENSE file for details.
 */

/**
 * \file matmult-oacc.c
 * \brief Matrix multiplication in C/OpenACC.
 * \author Sebastien Vincent
 * \date 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

#include <sys/time.h>

#include <openacc.h>

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
 * \return 0 if success, -1 if matrixes cannot be multiplied.
 */
int mat_mult_oacc(uint64_t* mat1, uint64_t* mat2, uint64_t* result, size_t m,
    size_t n, size_t w)
{
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  if(n != w)
  {
    return -1;
  }

  #pragma acc parallel copyin(mat1[0:(m * n)],mat2[0:(n * w)]) copyout(result[0:(n * w)])
  {
    #pragma acc loop independent
    for(i = 0 ; i < m ; i++)
    {
      #pragma acc loop independent
      for(j = 0 ; j < n ; j++)
      {
        uint64_t tmp = 0;

        #pragma acc loop independent reduction(+:tmp)
        for(k = 0 ; k < w ; k++)
        {
          tmp += mat1[i * w + k] * mat2[k * n + j];
        }

        result[i * m + j] = tmp;
      }
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
  fprintf(stdout, "Usage: %s [-m row size] [-t nb] "
      "[-p] [-h]\n\n"
      "  -h\t\tDisplay this help\n"
      "  -p\t\tPrint the input and output matrixes\n"
      "  -m row\tRow/column size (default 1024)\n", program);
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
   */
  static const char* options = "hpm:t:";
  int opt = 0;
  int print_matrix = 0;
  long m = DEFAULT_ROW_SIZE;
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
      default:
        fprintf(stderr, "Bad option (%c)\n", optopt);
        ret = -1;
        break;
    }
  }

  configuration->print_matrix = print_matrix;
  configuration->m = m;

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
  size_t nb_elements = 0;
  double start = 0;
  double end = 0;
  int ret = 0;
  int nb_devices = 0;

  nb_devices = acc_get_num_devices(acc_device_default);
  printf("Number of devices: %d\n", nb_devices);

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

  start = util_gettime_us();
  if(mat_mult_oacc(mat1, mat2, mat3, m, n, w) == -1)
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

