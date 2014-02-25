#ifndef VAR_TYPE
#define VAR_TYPE int
#endif  // VAR_TYPE

kernel void matrix_add(global const VAR_TYPE *A, global const VAR_TYPE *B,
                       global VAR_TYPE *C) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int cols = get_global_size(1);
  int idx = i * cols + j;
  C[idx] = A[idx] + B[idx];
}

kernel void matrix_sub(global const VAR_TYPE *A, global const VAR_TYPE *B,
                       global VAR_TYPE *C) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int cols = get_global_size(1);
  int idx = i * cols + j;
  C[idx] = A[idx] - B[idx];
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif  // BLOCK_SIZE

typedef enum { ROW, COL } DataDir;

typedef struct tag_matrix_param_t {
  int rows;
  int cols;
  DataDir dir;
} matrix_param_t;

inline VAR_TYPE matrix_get(global const VAR_TYPE* m, global const matrix_param_t* param,
                           int i, int j) {
  return param->dir == ROW ? m[i * param->cols + j] : m[j * param->rows + i];
}

kernel void matrix_mul(global const VAR_TYPE *A,
                       global const matrix_param_t *A_param,
                       global const VAR_TYPE *B,
                       global const matrix_param_t *B_param,
                       global VAR_TYPE *C) {
  int gx = get_group_id(0);
  int lx = get_local_id(0);
  int gy = get_group_id(1);
  int ly = get_local_id(1);

  local VAR_TYPE AS[BLOCK_SIZE][BLOCK_SIZE];
  local VAR_TYPE BS[BLOCK_SIZE][BLOCK_SIZE];

  int i_A, j_A, i_B, j_B;
  VAR_TYPE sum = 0;
  for (int j = 0, i = 0; j < A_param->cols; j += BLOCK_SIZE, i += BLOCK_SIZE) {
    j_A = j + lx;
    i_A = BLOCK_SIZE * gy + ly;
    j_B = BLOCK_SIZE * gx + lx;
    i_B = i + ly;
    // if current positions in shared matrices AS and BS are out of range then set 0
    AS[ly][lx] = (j_A < A_param->cols && i_A < A_param->rows) ?
        matrix_get(A, A_param, i_A, j_A) : 0;
    BS[ly][lx] = (j_B < B_param->cols && i_B < B_param->rows) ?
        matrix_get(B, B_param, i_B, j_B) : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      sum += AS[ly][k] * BS[k][lx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = sum;
}

#undef BLOCK_SIZE
#undef VAR_TYPE