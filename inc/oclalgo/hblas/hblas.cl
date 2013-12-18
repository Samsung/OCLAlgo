#ifndef VAR_TYPE
#define VAR_TYPE int
#endif  // VAR_TYPE

kernel void vector_add(global const VAR_TYPE *A, global const VAR_TYPE *B, 
                       global VAR_TYPE *C) {
	int i = get_global_id(0);
 	C[i] = A[i] + B[i];
}

kernel void matrix_add(global const VAR_TYPE *A, global const VAR_TYPE *B,  
                       global VAR_TYPE *C) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int cols = get_global_size(1);
  const int idx = i * cols + j;
  C[idx] = A[idx] + B[idx];
}

kernel void matrix_sub(global const VAR_TYPE *A, global const VAR_TYPE *B,  
                       global VAR_TYPE *C) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int cols = get_global_size(1);
  const int idx = i * cols + j;
  C[idx] = A[idx] - B[idx];
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif  // BLOCK_SIZE

kernel void matrix_mul(global const VAR_TYPE *A, global const VAR_TYPE *B, 
                       global VAR_TYPE *C, local VAR_TYPE *As, 
                       local VAR_TYPE *Bs, const int Aw, const int Bw) {
  int gx = get_group_id(0);
  int gy = get_group_id(1);
  int lx = get_local_id(0);
  int ly = get_local_id(1);

  int a_start = Aw * BLOCK_SIZE * gy;
  int a_end   = a_start + Aw - 1;
  int a_step  = BLOCK_SIZE;
  int b_start = BLOCK_SIZE * gx;
  int b_step  = BLOCK_SIZE * Bw;

  VAR_TYPE sum = 0;
  for (int a = a_start, b = b_start; a <= a_end; a += a_step, b += b_step) {
    As[ly * BLOCK_SIZE + lx] = A[a + Aw * ly + lx];
    Bs[ly * BLOCK_SIZE + lx] = B[b + Bw * ly + lx];
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k)
      sum += As[ly * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + lx];
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = sum;
}

#undef BLOCK_SIZE
#undef VAR_TYPE