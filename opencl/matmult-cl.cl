/*
 * Copyright (c) 2014-2016, Sebastien Vincent
 *
 * Distributed under the terms of the BSD 3-clause License.
 * See the LICENSE file for details.
 */

/**
 * \file matmult-cl.cl
 * \brief OpenCL matrix multiplication.
 * \author Sebastien Vincent
 * \date 2014-2016
 */

/**
 * \def BLOCK_SIZE
 * \brief Size of a matrix block.
 */
#define BLOCK_SIZE 16

/**
 * \brief Multiply two matrixes and store result in third ones.
 * \param mat1 first matrix.
 * \param mat2 second matrix.
 * \param result result matrix.
 * \param M row size of first matrix.
 * \param N column size of first matrix.
 * \param W row size of second matrix.
 */
__kernel void matmult(__global int* mat1, __global int* mat2,
    __global int* result, int M, int N, int W)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int tmp = 0;

  for(size_t k = 0 ; k < W ; k++)
  {
    tmp += mat1[i * W + k] * mat2[k * N + j];
  }

  result[i * M + j] = tmp;
}

/**
 * \brief Optimized multiplication of two matrixes.
 * \param mat1 first matrix.
 * \param mat2 second matrix.
 * \param result result matrix.
 * \param M row size of first matrix.
 * \param N column size of first matrix.
 * \param W row size of second matrix.
 */
__kernel void matmult2(__global int* mat1, __global int* mat2,
    __global int* result, int M, int N, int W)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int groupi = get_group_id(0);
    int groupj = get_group_id(1);
    int loci = get_local_id(0);
    int locj = get_local_id(1);
    /* row group starts at row block (row size x block size) x row number
     * column group starts at block size x column number
     */
    size_t mat1_offset = M * BLOCK_SIZE * groupi;
    size_t mat1_offset_end = mat1_offset + M;
    size_t mat2_offset = BLOCK_SIZE * groupj;
    size_t mat1_step = BLOCK_SIZE;
    size_t mat2_step = BLOCK_SIZE * W;
    int tmp = 0;

    for(size_t off1 = mat1_offset, off2 = mat2_offset ; off1 < mat1_offset_end ;
        off1 += mat1_step, off2 += mat2_step)
    {
        __local int local_row[BLOCK_SIZE * BLOCK_SIZE];
        __local int local_col[BLOCK_SIZE * BLOCK_SIZE];
        size_t idx = loci * BLOCK_SIZE + locj;

        /* copy block of matrix from global memory to local */
        local_row[idx] = mat1[off1 + loci * M + locj];
        local_col[idx] = mat2[off2 + locj * W + loci];

        /* wait until all data are copied to local memory */
        barrier(CLK_LOCAL_MEM_FENCE);

        for(size_t k = 0 ; k < BLOCK_SIZE ; k++)
        {
            /* multiply */
            tmp += local_row[loci * BLOCK_SIZE + k] *
              local_col[locj * BLOCK_SIZE + k];
        }

        /* wait until the full row has been calculated */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[i * M + j] = tmp;
}

/**
 * \brief Optimized multiplication of two matrixes.
 * \param mat1 first matrix.
 * \param mat2 second matrix.
 * \param result result matrix.
 * \param M row size of first matrix.
 * \param N column size of first matrix.
 * \param W row size of second matrix.
 */
__kernel void matmult3(__global int* mat1, __global int* mat2,
    __global int* result, int M, int N, int W)
{
    int groupi = get_group_id(0);
    int groupj = get_group_id(1);
    int loci = get_local_id(0);
    int locj = get_local_id(1);
    /*
     * row matrix starts at row block (row size x block size) x row number
     * column matrix starts at block size x column number
     */
    size_t mat1_offset = M * BLOCK_SIZE * groupi;
    size_t mat1_offset_end = mat1_offset + M;
    size_t mat2_offset = BLOCK_SIZE * groupj;
    size_t mat1_step = BLOCK_SIZE;
    size_t mat2_step = BLOCK_SIZE * W;
    int tmp = 0;

    for(size_t off1 = mat1_offset, off2 = mat2_offset ; off1 < mat1_offset_end ;
        off1 += mat1_step, off2 += mat2_step)
    {
        __local int local_row[BLOCK_SIZE][BLOCK_SIZE];
        __local int local_col[BLOCK_SIZE][BLOCK_SIZE];

        /* copy block of matrix from global memory to local */
        local_row[locj][loci] = mat1[off1 + locj * M + loci];
        local_col[locj][loci] = mat2[off2 + locj * W + loci];

        /* wait until all data are copied to local memory */
        barrier(CLK_LOCAL_MEM_FENCE);

        for(size_t k = 0 ; k < BLOCK_SIZE ; k++)
        {
            /* multiply */
            tmp += local_row[loci][k] * local_col[k][locj];
        }

        /* wait until the full row has been calculated to perform another one */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[get_global_id(0) * M + get_global_id(1)] = tmp;
}

