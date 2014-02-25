kernel void vector_add(global const int *A, global const int *B, global int *C) {
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
}