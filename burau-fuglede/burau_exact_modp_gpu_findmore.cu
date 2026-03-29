#define main burau_exact_modp_gpu_singlehit_main
#include "burau_exact_modp_gpu.cu"
#undef main

#ifndef MAX_HITS_PER_LEVEL
#define MAX_HITS_PER_LEVEL 1048576
#endif

__global__ void exact_level_collect_kernel(
  int level,
  const Tuple5* survivors,
  int survivor_count,
  HitRecord* hits,
  int max_hits,
  int* hit_count,
  int* hit_overflow_flag
)
{
  int idx = blockIdx.x;
  extern __shared__ unsigned char smem[];
  int* keys = (int*)smem;
  unsigned char* vals = (unsigned char*)(keys + HASH_CAPACITY);

  if(idx >= survivor_count)
    return;

  if(threadIdx.x == 0)
  {
    Tuple5 t = survivors[idx];
    if(exact_zero_sparse(t, keys, vals))
    {
      int slot = atomicAdd(hit_count, 1);
      if(slot < max_hits)
      {
        hits[slot].level = level;
        hits[slot].tuple = t;
      }
      else
      {
        *hit_overflow_flag = 1;
      }
    }
  }
}

int main(int argc, char** argv)
{
  int start_level = START_LEVEL;
  int stop_level = STOP_LEVEL;
  int level;
  Tuple5* d_survivors = NULL;
  int* d_survivor_count = NULL;
  int* d_overflow_flag = NULL;
  int* d_found_flag = NULL;
  HitRecord* d_hits = NULL;
  int* d_hit_count = NULL;
  int* d_hit_overflow_flag = NULL;
  int h_overflow = 0;
  int h_survivors = 0;
  int h_hit_count = 0;
  int h_hit_overflow = 0;
  uint64_t total_hits = 0;
  size_t shared_bytes =
    HASH_CAPACITY * sizeof(int) + HASH_CAPACITY * sizeof(unsigned char);

  if(argc >= 2)
    start_level = atoi(argv[1]);
  if(argc >= 3)
    stop_level = atoi(argv[2]);

  if(stop_level > MAX_LEVEL)
  {
    fprintf(stderr,
      "stop_level=%d exceeds MAX_LEVEL=%d; recompile with larger MAX_LEVEL\n",
      stop_level, MAX_LEVEL);
    return 1;
  }

  cuda_check(cudaMalloc(&d_survivors,
    (size_t)MAX_SURVIVORS_PER_LEVEL * sizeof(Tuple5)),
    "cudaMalloc survivors");
  cuda_check(cudaMalloc(&d_survivor_count, sizeof(int)),
    "cudaMalloc survivor_count");
  cuda_check(cudaMalloc(&d_overflow_flag, sizeof(int)),
    "cudaMalloc overflow_flag");
  cuda_check(cudaMalloc(&d_found_flag, sizeof(int)),
    "cudaMalloc found_flag");
  cuda_check(cudaMalloc(&d_hits,
    (size_t)MAX_HITS_PER_LEVEL * sizeof(HitRecord)),
    "cudaMalloc hits");
  cuda_check(cudaMalloc(&d_hit_count, sizeof(int)),
    "cudaMalloc hit_count");
  cuda_check(cudaMalloc(&d_hit_overflow_flag, sizeof(int)),
    "cudaMalloc hit_overflow_flag");

  cuda_check(cudaMemset(d_found_flag, 0, sizeof(int)), "memset found_flag");

  for(level = start_level; level <= stop_level; level++)
  {
    uint64_t total =
      ((uint64_t)level * (uint64_t)(level + 1) * (uint64_t)(level + 1)) / 2ULL;
    int blocks = (int)((total + THREADS_PER_BLOCK - 1ULL) / THREADS_PER_BLOCK);

    if(blocks > 65535)
      blocks = 65535;
    if(blocks < 1)
      blocks = 1;

    cuda_check(cudaMemset(d_survivor_count, 0, sizeof(int)),
      "memset survivor_count");
    cuda_check(cudaMemset(d_overflow_flag, 0, sizeof(int)),
      "memset overflow_flag");
    cuda_check(cudaMemset(d_hit_count, 0, sizeof(int)),
      "memset hit_count");
    cuda_check(cudaMemset(d_hit_overflow_flag, 0, sizeof(int)),
      "memset hit_overflow_flag");

    screen_level_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      level,
      d_survivors,
      MAX_SURVIVORS_PER_LEVEL,
      d_survivor_count,
      d_overflow_flag,
      d_found_flag
    );
    cuda_check(cudaGetLastError(), "launch screen_level_kernel");
    cuda_check(cudaDeviceSynchronize(), "sync screen_level_kernel");

    cuda_check(cudaMemcpy(&h_overflow, d_overflow_flag, sizeof(int),
      cudaMemcpyDeviceToHost), "copy overflow_flag");
    cuda_check(cudaMemcpy(&h_survivors, d_survivor_count, sizeof(int),
      cudaMemcpyDeviceToHost), "copy survivor_count");

    if(h_overflow)
    {
      fprintf(stderr,
        "survivor buffer overflow at level %d; increase MAX_SURVIVORS_PER_LEVEL\n",
        level);
      return 1;
    }

    if(h_survivors > 0)
    {
      exact_level_collect_kernel<<<h_survivors, 1, shared_bytes>>>(
        level,
        d_survivors,
        h_survivors,
        d_hits,
        MAX_HITS_PER_LEVEL,
        d_hit_count,
        d_hit_overflow_flag
      );
      cuda_check(cudaGetLastError(), "launch exact_level_collect_kernel");
      cuda_check(cudaDeviceSynchronize(), "sync exact_level_collect_kernel");
    }

    cuda_check(cudaMemcpy(&h_hit_count, d_hit_count, sizeof(int),
      cudaMemcpyDeviceToHost), "copy hit_count");
    cuda_check(cudaMemcpy(&h_hit_overflow, d_hit_overflow_flag, sizeof(int),
      cudaMemcpyDeviceToHost), "copy hit_overflow_flag");

    if(h_hit_overflow)
    {
      fprintf(stderr,
        "hit buffer overflow at level %d; increase MAX_HITS_PER_LEVEL\n",
        level);
      return 1;
    }

    total_hits += (uint64_t)h_hit_count;

    printf("level=%d candidates=%llu survivors=%d hits=%d total_hits=%llu\n",
      level,
      (unsigned long long)total,
      h_survivors,
      h_hit_count,
      (unsigned long long)total_hits);
    fflush(stdout);

    if(h_hit_count > 0)
    {
      HitRecord* h_hits = (HitRecord*)malloc((size_t)h_hit_count * sizeof(HitRecord));
      int i;

      if(h_hits == NULL)
      {
        fprintf(stderr, "host malloc failed for %d hits at level %d\n",
          h_hit_count, level);
        return 1;
      }

      cuda_check(cudaMemcpy(h_hits, d_hits, (size_t)h_hit_count * sizeof(HitRecord),
        cudaMemcpyDeviceToHost), "copy hits");

      for(i = 0; i < h_hit_count; i++)
      {
        printf("HIT level=%d tuple=(%d,%d,%d,%d,%d)\n",
          h_hits[i].level,
          h_hits[i].tuple.a,
          h_hits[i].tuple.b,
          h_hits[i].tuple.c,
          h_hits[i].tuple.d,
          h_hits[i].tuple.e);
      }
      fflush(stdout);
      free(h_hits);
    }
  }

  cudaFree(d_survivors);
  cudaFree(d_survivor_count);
  cudaFree(d_overflow_flag);
  cudaFree(d_found_flag);
  cudaFree(d_hits);
  cudaFree(d_hit_count);
  cudaFree(d_hit_overflow_flag);

  return 0;
}
