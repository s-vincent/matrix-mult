/*
 * Copyright (c) 2014-2016, Sebastien Vincent
 *
 * Distributed under the terms of the BSD 3-clause License.
 * See the LICENSE file for details.
 */

/**
 * \file matmult-cl.c
 * \brief Matrix multiplication in C/OpenCL.
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

#include "util_opencl.h"

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
void mat_init(cl_ulong* mat1, cl_ulong* mat2, size_t m, size_t n)
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
void mat_print(cl_ulong* mat, size_t m, size_t n)
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
 * \brief Performs multiplication of matrixes using OpenCL.
 * \param mat1 first matrix.
 * \param mat2 second matrix.
 * \param result result matrix.
 * \param M row size of first matrix.
 * \param N column size of first matrix.
 * \param W row size of second matrix.
 * \return 0 if success, -1 if matrixes cannot be multiplied or some OpenCL
 * blocking errors.
 */
int mat_mult_cl(cl_ulong* mat1, cl_ulong* mat2, cl_ulong* result, size_t M,
    size_t N, size_t W)
{
  int ret = 0;
  cl_platform_id* platforms = NULL;
  cl_int status = CL_SUCCESS;
  int nb_platforms = 0;
  double start = 0;
  double end = 0;
  int success = 0;

  if((nb_platforms = opencl_get_platforms(&platforms, &status)) <= 0)
  {
    fprintf(stderr, "No platforms: nb_platforms=%d status=%d\n", nb_platforms,
        status);
    return -1;
  }

  for(int i = 0 ; i < nb_platforms ; i++)
  {
    cl_device_id* devices = NULL;
    int nb_devices = 0;
    cl_context context;
    cl_program program;
    cl_kernel* kernels = NULL;
    int nb_kernels = 0;
    cl_mem input_mat1;
    cl_mem input_mat2;
    cl_mem output_result;
    cl_context_properties context_props[] =
      {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0};

    if((nb_devices = opencl_get_devices(platforms[i], &devices,
            CL_DEVICE_TYPE_ALL, &status)) <= 0)
    {
      fprintf(stderr, "No devices: nb_devices=%d status=%d\n", nb_devices,
          status);
      continue;
    }

    fprintf(stdout, "Platform %d has %d device(s)\n", i, nb_devices);

    context = clCreateContext(context_props, nb_devices, devices, NULL, NULL,
        &status);

    if(status != CL_SUCCESS)
    {
      fprintf(stderr, "Failed to create context: status=%d\n", status);
      free(devices);
      continue;
    }

    /* get the OpenCL program from the file and build it */
    if((ret = opencl_get_program_from_file(context, "./matmult-cl.cl", &program,
          &status)) != 0)
    {
      fprintf(stderr, "opencl_get_program_from_file: error:%d status=%d\n",
          ret, status);

      clReleaseContext(context);
      free(devices);
      continue;
    }

    if((status = clBuildProgram(program, nb_devices, devices, NULL, NULL,
            NULL)) != CL_SUCCESS)
    {
      cl_build_status build_status;

      fprintf(stderr, "clBuildProgram failed: error:%d status=%d\n",
          ret, status);

      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_STATUS,
          sizeof(cl_build_status), &build_status, NULL);

      if(build_status == CL_BUILD_ERROR)
      {
        char* log = NULL;
        size_t log_size = 0;

        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0,
            NULL, &log_size);
        log = malloc(sizeof(char) * log_size);

        if(log)
        {
          clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
              log_size, log, NULL);
          log[log_size] = 0x00;

          fprintf(stderr, "Build error log:\n%s\n", log);
          free(log);
        }
        else
        {
          fprintf(stderr, "Cannot print build log error.\n");
        }
      }

      clReleaseProgram(program);
      clReleaseContext(context);
      free(devices);
      continue;
    }

    /* retrieves kernels from program */
    if((nb_kernels = opencl_get_kernels(program, &kernels, &status)) <= 0)
    {
      fprintf(stderr, "No kernels found: nb_kernels=%d status=%d\n",
          nb_kernels, status);

      clReleaseProgram(program);
      clReleaseContext(context);
      free(devices);
      continue;
    }

    /* try to create a queue for one device */
    for(int di = 0 ; di < nb_devices ; di++)
    {
      cl_device_id device = devices[di];
      cl_command_queue queue;
      char device_name[1024];

#if CL_TARGET_OPENCL_VERSION < 200
      queue = clCreateCommandQueue(context, device, 0, &status);
#else
      queue = clCreateCommandQueueWithProperties(context, device,
          NULL, &status);
#endif

      if(status != CL_SUCCESS)
      {
        printf("clCreateCommandQueue failed on device %d platform %d\n", di, i);
        continue;
      }

      clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name,
          NULL);

      /* creates the different OpenCL buffer */
      input_mat1 = clCreateBuffer(context,
          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * N * sizeof(cl_ulong),
          mat1, &status);
      input_mat2 = clCreateBuffer(context,
          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, W * N * sizeof(cl_ulong),
          mat2, &status);
      output_result = clCreateBuffer(context,
          CL_MEM_WRITE_ONLY, W * M * sizeof(cl_ulong), NULL, &status);

      /* execute all the kernels */
      for(int ki = 0 ; ki < nb_kernels ; ki++)
      {
        char kernel_name[1024];
        size_t global_work_offset[2] = {0, 0};
        size_t global_work_size[2] = {M, N};
        size_t local_work_size[2] = {16, 16};

        clGetKernelInfo(kernels[ki], CL_KERNEL_FUNCTION_NAME,
            sizeof(kernel_name), kernel_name, NULL);

        status = clSetKernelArg(kernels[ki], 0, sizeof(cl_mem), &input_mat1);
        status |= clSetKernelArg(kernels[ki], 1, sizeof(cl_mem), &input_mat2);
        status |= clSetKernelArg(kernels[ki], 2, sizeof(cl_mem),
            &output_result);
        status |= clSetKernelArg(kernels[ki], 3, sizeof(cl_uint), &M);
        status |= clSetKernelArg(kernels[ki], 4, sizeof(cl_uint), &N);
        status |= clSetKernelArg(kernels[ki], 5, sizeof(cl_uint), &W);

        if(status != CL_SUCCESS)
        {
          fprintf(stderr,
              "Failed to set arguments for kernel %s on %s: status=%d\n",
              kernel_name, device_name, status);
          continue;
        }

        start = util_gettime_us();
        if((status = clEnqueueNDRangeKernel(queue, kernels[ki], 2,
                global_work_offset, global_work_size, local_work_size,
                0, NULL, NULL)) != CL_SUCCESS)
        {
          fprintf(stderr, "Failed to clEnqueueTask %s on %s: status=%d\n",
              kernel_name, device_name, status);
          continue;
        }

        if((status = clEnqueueReadBuffer(queue, output_result, CL_FALSE, 0,
                M * N * sizeof(cl_ulong), result, 0, NULL, NULL)) != CL_SUCCESS)
        {
          fprintf(stderr,
              "clEnqueueReadBuffer failed for kernel %s on %s: status=%d\n",
              kernel_name, device_name, status);
          continue;
        }

        clFinish(queue);
        end = util_gettime_us();
        success = 1;
        fprintf(stdout, "\t%s executed on %s in \t%f ms\n", kernel_name,
            device_name, (end - start) / 1000);
      }

      clReleaseMemObject(input_mat1);
      clReleaseMemObject(input_mat2);
      clReleaseMemObject(output_result);
      clReleaseCommandQueue(queue);
    }

    opencl_release_kernels(&kernels, nb_kernels);
    clReleaseProgram(program);
    clReleaseContext(context);
    free(devices);
  }

  free(platforms);
  return success ? 0 : -1;
}

/**
 * \brief Print help.
 * \param program program name.
 */
void print_help(const char* program)
{
  fprintf(stdout, "Usage: %s [-m row size] [-p] [-h]\n\n"
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
   */
  static const char* options = "hpm:";
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
  cl_ulong* mat1 = NULL;
  cl_ulong* mat2 = NULL;
  cl_ulong* mat3 = NULL;
  size_t m = DEFAULT_ROW_SIZE;
  size_t n = DEFAULT_COLUMN_SIZE;
  size_t w = DEFAULT_COLUMN_SIZE;
  int print_matrix = 0;
  struct configuration config;
  int nb_elements = 0;
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

  nb_elements = m * n;

  mat1 = malloc(nb_elements * sizeof(cl_ulong));
  mat2 = malloc(nb_elements * sizeof(cl_ulong));
  mat3 = malloc(nb_elements * sizeof(cl_ulong));

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

  if(mat_mult_cl(mat1, mat2, mat3, m, n, w) == -1)
  {
    fprintf(stderr, "Matrixes cannot be multiplied\n");
    ret = EXIT_FAILURE;
  }
  else
  {
    fprintf(stdout, "Multiplication success\n");

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

