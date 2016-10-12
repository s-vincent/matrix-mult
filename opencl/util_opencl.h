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
 * \file util_opencl.h
 * \brief OpenCL utility.
 * \author Sebastien Vincent
 * \date 2014
 */

#ifndef VS_UTIL_OPENCL_H
#define VS_UTIL_OPENCL_H

#include <CL/cl.h>

/**
 * \brief Retrieves OpenCL platforms.
 * \param platforms pointer that will handle platforms (dynamically allocated).
 * \param status OpenCL last error code (useful if return code equals -1).
 * \return negative integer if failure, number of platforms otherwise.
 * \note Caller MUST free *platforms after use if return code is greater than 0.
 */
int opencl_get_platforms(cl_platform_id** platforms, cl_int* status);

/**
 * \brief Retrieves OpenCL devices for a specific OpenCL platform.
 * \param platform the OpenCL platform.
 * \param devices pointer that will handle devices (dynamically allocated).
 * \param type OpenCL device type to return.
 * \param status OpenCL last error code (useful if return code equals -1).
 * \return negative integer if failure, number of devices otherwise.
 * \note Caller MUST free *devices after use if return code is greater than 0.
 */
int opencl_get_devices(cl_platform_id platform, cl_device_id** devices,
  cl_device_type type, cl_int* status);

/**
 * \brief Retrieves OpenCL program from a file.
 * \param context OpenCL context to use.
 * \param file_path path of the OpenCL file.
 * \param program pointer that will handle OpenCL program.
 * \param status OpenCL last error code (useful if return code equals -1).
 * \return 0 if success, negative integer (errno) otherwise.
 */
int opencl_get_program_from_file(cl_context context, const char* file_path,
    cl_program* program, cl_int* status);

/**
 * \brief Retrieves OpenCL file content.
 * \param file_path path of the OpenCL file.
 * \param program pointer that will handle file content (dynamically allocated).
 * \param program_size size of the file content.
 * \return 0 if success, negative integer (errno) otherwise.
 * \note Caller MUST free *program after use if return code is 0.
 */
int opencl_get_file_data(const char* file_path, char** program,
  size_t* program_size);

/**
 * \brief Retrieves all kernels in an OpenCL program.
 * \param program OpenCL program.
 * \param kernels pointer that will handle kernels (dynamically allocated).
 * \param status OpenCL last error code (userful if return code equals -1).
 * \return negative integer if failure, number of kernels otherwise.
 * \note Caller MUST use opencl_release_kernels with kernels after use
 * if return code is greater than 0.
 */
int opencl_get_kernels(cl_program program, cl_kernel** kernels, cl_int* status);

/**
 * \brief Releases and frees all the kernels.
 * \param kernels pointer containing OpenCL kernels to release.
 * \param nb number of kernels.
 */
void opencl_release_kernels(cl_kernel** kernels, size_t nb);

#endif /* VS_UTIL_OPENCL_H */

