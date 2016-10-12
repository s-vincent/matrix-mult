/*
 * Copyright (C) 2014 Sebastien Vincent.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

/**
 * \file util_opencl.c
 * \brief OpenCL utility.
 * \author Sebastien Vincent
 * \date 2014
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "util_opencl.h"

int opencl_get_platforms(cl_platform_id** platforms, cl_int* status)
{
  cl_uint nb = 0;
  int ret = -1;

  *platforms = NULL;

  if((*status = clGetPlatformIDs(0, NULL, &nb)) == CL_SUCCESS)
  {
    if(nb > 0)
    {
      *platforms = malloc(sizeof(cl_platform_id) * nb);

      if(*platforms)
      {
        if((*status = clGetPlatformIDs(nb, *platforms, NULL)) == CL_SUCCESS)
        {
          ret = (int)nb;
        }
      }
      else
      {
        ret = -errno;
      }
    }
    else
    {
      ret = 0;
    }
  }

  return ret;
}

int opencl_get_devices(cl_platform_id platform, cl_device_id** devices,
    cl_device_type type,  cl_int* status)
{
  cl_uint nb = 0;
  int ret = -1;

  *devices = NULL;

  if((*status = clGetDeviceIDs(platform, type, 0, NULL, &nb)) == CL_SUCCESS)
  {
    if(nb > 0)
    {
      *devices = malloc(sizeof(cl_device_id) * nb);

      if(*devices)
      {
        if((*status = clGetDeviceIDs(platform, type, nb, *devices, NULL))
            == CL_SUCCESS)
        {
          ret = (int)nb;
        }
      }
      else
      {
        ret = -errno;
      }
    }
    else
    {
      ret = 0;
    }
  }

  return ret;
}

int opencl_get_program_from_file(cl_context context, const char* file_path,
    cl_program* program, cl_int* status)
{
  int ret = 0;
  size_t data_size = 0;
  char* data = NULL;

  *status = CL_SUCCESS;
  *program = NULL;

  ret = opencl_get_file_data(file_path, &data, & data_size);

  if(ret < 0)
  {
    return ret;
  }

  *program = clCreateProgramWithSource(context, 1, (const char**)&data,
      &data_size, status);
  free(data);

  return ret;
}

int opencl_get_file_data(const char* file_path, char** program,
  size_t* program_size)
{
  char* ret = NULL;
  size_t file_size = 0;
  size_t n = 0;
  size_t nb = 0;
  FILE* file = fopen(file_path, "r");

  *program = NULL;
  *program_size = 0;

  if(!file)
  {
    return -errno;
  }

  /* get the size of the content */
  fseek(file, 0, SEEK_END);
  file_size = ftell(file);
  rewind(file);

  ret = malloc(sizeof(char) * file_size + 1);

  if(!ret)
  {
    fclose(file);
    return -errno;
  }

  while((n = fread(ret, sizeof(char), file_size, file)) > 0)
  {
    nb += n;
  }

  ret[nb] = 0x00;

  if(nb != file_size)
  {
    free(ret);
    return -errno;
  }

  fclose(file);
  *program = ret;
  *program_size = file_size;
  return 0;
}

int opencl_get_kernels(cl_program program, cl_kernel** kernels, cl_int* status)
{
  cl_uint nb = 0;
  int ret = -1;

  *kernels = NULL;

  if((*status = clCreateKernelsInProgram(program, 0, NULL, &nb)) == CL_SUCCESS)
  {
    if(nb > 0)
    {
      *kernels = malloc(sizeof(cl_kernel) * nb);

      if(*kernels)
      {
        if((*status = clCreateKernelsInProgram(program, nb, *kernels, NULL))
            == CL_SUCCESS)
        {
          ret = (int)nb;
        }
      }
      else
      {
        ret = -errno;
      }
    }
    else
    {
      ret = 0;
    }
  }

  return ret;
}

void opencl_release_kernels(cl_kernel** kernels, size_t nb)
{
  for(size_t i = 0 ; i < nb ; i++)
  {
    clReleaseKernel((*kernels)[i]);
  }

  free((*kernels));
  *kernels = NULL;
}

