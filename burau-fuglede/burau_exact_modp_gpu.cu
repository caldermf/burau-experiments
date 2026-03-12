#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * GPU-oriented exact mod-p Burau search.
 *
 * Design goals:
 *   - No sequential increment() on device. Candidates are indexed directly by
 *     level L = d + e and ordinal within that level.
 *   - Very cheap first-stage necessary filter: the pairing must vanish at every
 *     q in F_p. We fuse all q-values into one walk.
 *   - Exact polynomial test does not store a dense coefficient array and never
 *     shifts it. Instead, it tracks:
 *         poly(t) = t^S * sum_i c_i t^{k_i}
 *     and records only the event exponents k_i = mon_exp - S.
 *     A common final shift S is irrelevant for deciding poly == 0.
 *   - The exact stage is run only on survivors from the cheap filter.
 *
 * This is written for CUDA. The current environment does not have nvcc or a
 * CUDA runtime, so this file is not compiled here.
 */

#ifndef PRIME
#define PRIME 3
#endif

#ifndef START_LEVEL
#define START_LEVEL 1
#endif

#ifndef STOP_LEVEL
#define STOP_LEVEL 512
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef MAX_LEVEL
#define MAX_LEVEL 2048
#endif

#ifndef MAX_SURVIVORS_PER_LEVEL
#define MAX_SURVIVORS_PER_LEVEL 1048576
#endif

/* The exact stage only ever sees survivors, so a moderate shared-memory hash
 * is a good tradeoff. */
#define HASH_CAPACITY (2 * MAX_LEVEL + 32)
#define EMPTY_KEY INT32_MIN

struct Tuple5 {
  int a;
  int b;
  int c;
  int d;
  int e;
};

struct HitRecord {
  int level;
  Tuple5 tuple;
};

static inline void cuda_check(cudaError_t err, const char* msg)
{
  if(err != cudaSuccess)
  {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}

__device__ __forceinline__ int positive_mod_p(int x)
{
  x %= PRIME;
  if(x < 0)
    x += PRIME;
  return x;
}

__device__ __forceinline__ uint32_t mix32(uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ uint64_t triples_before_row(int level, int a)
{
  /* Number of triples (a',b,c) with a' < a and a'+b+c = level-1. */
  return ((uint64_t)a * (2ULL * (uint64_t)level - (uint64_t)a + 1ULL)) / 2ULL;
}

__device__ __forceinline__ Tuple5 decode_candidate(int level, uint64_t ordinal)
{
  Tuple5 t;
  uint64_t abc_index = ordinal / (uint64_t)(level + 1);
  uint64_t row_start;
  double disc;
  int a;

  t.d = (int)(ordinal % (uint64_t)(level + 1));
  t.e = level - t.d;

  /*
   * Rows have lengths:
   *   level, level-1, ..., 1
   * corresponding to a = 0,1,...,level-1
   */
  disc = (double)(2 * level + 1);
  disc = disc * disc - 8.0 * (double)abc_index;
  a = (int)floor(((double)(2 * level + 1) - sqrt(disc)) * 0.5);
  if(a < 0)
    a = 0;
  if(a >= level)
    a = level - 1;

  row_start = triples_before_row(level, a);
  while(a > 0 && row_start > abc_index)
  {
    a--;
    row_start = triples_before_row(level, a);
  }
  while(a + 1 < level && triples_before_row(level, a + 1) <= abc_index)
  {
    a++;
    row_start = triples_before_row(level, a);
  }

  t.a = a;
  t.b = (int)(abc_index - row_start);
  t.c = level - 1 - t.a - t.b;
  return t;
}

__device__ __forceinline__ bool passes_field_filter(const Tuple5& t)
{
  int bl = 2 * t.a;
  int start = bl + t.b;
  int cl = start + t.b + 1;
  int end = cl + t.c;
  int el = 2 * t.d;
  int er = el + t.e;
  int suma = bl - 1;
  int sumb = 2 * start;
  int sumc = 2 * end;
  int sumd = el - 1;
  int sume = er + er - 1;
  int x = start;
  int togo = t.d + t.e;
  int poly[PRIME];
  int mon[PRIME];
  int q4[PRIME];
  int q;

#pragma unroll
  for(q = 0; q < PRIME; q++)
  {
    poly[q] = 0;
    mon[q] = 1;
    q4[q] = positive_mod_p(q * q);
    q4[q] = positive_mod_p(q4[q] * q4[q]);
  }

  while(1)
  {
    if(x < el)
    {
      if(x < t.d)
      {
#pragma unroll
        for(q = 0; q < PRIME; q++)
        {
          mon[q] = positive_mod_p(mon[q] * q);
          poly[q] = positive_mod_p(poly[q] + mon[q]);
        }
      }
      else
      {
#pragma unroll
        for(q = 0; q < PRIME; q++)
        {
          poly[q] = positive_mod_p(poly[q] - mon[q]);
          poly[q] = positive_mod_p(poly[q] * q);
        }
      }
      x = sumd - x;
    }
    else
    {
      if(x < er)
      {
#pragma unroll
        for(q = 0; q < PRIME; q++)
        {
          poly[q] = positive_mod_p(poly[q] - mon[q]);
          mon[q] = positive_mod_p(mon[q] * q);
        }
      }
      else
      {
#pragma unroll
        for(q = 0; q < PRIME; q++)
        {
          poly[q] = positive_mod_p(poly[q] * q);
          poly[q] = positive_mod_p(poly[q] + mon[q]);
        }
      }
      x = sume - x;
    }

    togo--;

    if(x < cl)
    {
      if(x < bl)
      {
        if(x < t.a)
        {
#pragma unroll
          for(q = 0; q < PRIME; q++)
            poly[q] = positive_mod_p(poly[q] * q);
        }
        else
        {
#pragma unroll
          for(q = 0; q < PRIME; q++)
            mon[q] = positive_mod_p(mon[q] * q);
        }
        x = suma - x;
      }
      else
      {
        if(x < start)
        {
#pragma unroll
          for(q = 0; q < PRIME; q++)
            mon[q] = positive_mod_p(mon[q] * q4[q]);
        }
        else
        {
#pragma unroll
          for(q = 0; q < PRIME; q++)
            poly[q] = positive_mod_p(poly[q] * q4[q]);
        }
        x = sumb - x;
      }
    }
    else
    {
      if(x < end)
      {
#pragma unroll
        for(q = 0; q < PRIME; q++)
          poly[q] = positive_mod_p(poly[q] * q);
      }
      else if(x > end)
      {
#pragma unroll
        for(q = 0; q < PRIME; q++)
          mon[q] = positive_mod_p(mon[q] * q);
      }
      else
      {
        if(togo != 0)
          return false;

#pragma unroll
        for(q = 0; q < PRIME; q++)
          if(poly[q] != 0)
            return false;

        return true;
      }
      x = sumc - x;
    }
  }
}

__global__ void screen_level_kernel(
  int level,
  Tuple5* survivors,
  int max_survivors,
  int* survivor_count,
  int* overflow_flag,
  int* found_flag
)
{
  uint64_t total =
    ((uint64_t)level * (uint64_t)(level + 1) * (uint64_t)(level + 1)) / 2ULL;
  uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;

  while(idx < total)
  {
    Tuple5 t;
    int slot;

    if(*found_flag)
      return;

    t = decode_candidate(level, idx);
    if(passes_field_filter(t))
    {
      slot = atomicAdd(survivor_count, 1);
      if(slot < max_survivors)
        survivors[slot] = t;
      else
        *overflow_flag = 1;
    }
    idx += stride;
  }
}

__device__ __forceinline__ void hash_add(
  int key,
  int delta,
  int* keys,
  unsigned char* vals
)
{
  uint32_t pos = mix32((uint32_t)key) % HASH_CAPACITY;

  while(1)
  {
    if(keys[pos] == EMPTY_KEY)
    {
      keys[pos] = key;
      vals[pos] = (unsigned char)positive_mod_p(delta);
      return;
    }
    if(keys[pos] == key)
    {
      vals[pos] = (unsigned char)positive_mod_p((int)vals[pos] + delta);
      return;
    }
    pos++;
    if(pos == HASH_CAPACITY)
      pos = 0;
  }
}

__device__ bool exact_zero_sparse(const Tuple5& t, int* keys, unsigned char* vals)
{
  int bl = 2 * t.a;
  int start = bl + t.b;
  int cl = start + t.b + 1;
  int end = cl + t.c;
  int el = 2 * t.d;
  int er = el + t.e;
  int suma = bl - 1;
  int sumb = 2 * start;
  int sumc = 2 * end;
  int sumd = el - 1;
  int sume = er + er - 1;
  int x = start;
  int mon_exp = 0;
  int poly_shift = 0;
  int i;

  for(i = 0; i < HASH_CAPACITY; i++)
  {
    keys[i] = EMPTY_KEY;
    vals[i] = 0;
  }

  while(1)
  {
    if(x < el)
    {
      if(x < t.d)
      {
        mon_exp++;
        hash_add(mon_exp - poly_shift, +1, keys, vals);
      }
      else
      {
        hash_add(mon_exp - poly_shift, -1, keys, vals);
        poly_shift++;
      }
      x = sumd - x;
    }
    else
    {
      if(x < er)
      {
        hash_add(mon_exp - poly_shift, -1, keys, vals);
        mon_exp++;
      }
      else
      {
        poly_shift++;
        hash_add(mon_exp - poly_shift, +1, keys, vals);
      }
      x = sume - x;
    }

    if(x < cl)
    {
      if(x < bl)
      {
        if(x < t.a)
          poly_shift++;
        else
          mon_exp++;
        x = suma - x;
      }
      else
      {
        if(x < start)
          mon_exp += 4;
        else
          poly_shift += 4;
        x = sumb - x;
      }
    }
    else
    {
      if(x < end)
        poly_shift++;
      else if(x > end)
        mon_exp++;
      else
      {
        for(i = 0; i < HASH_CAPACITY; i++)
          if(vals[i] != 0)
            return false;
        return true;
      }
      x = sumc - x;
    }
  }
}

__global__ void exact_level_kernel(
  int level,
  const Tuple5* survivors,
  int survivor_count,
  HitRecord* hit,
  int* found_flag
)
{
  int idx = blockIdx.x;
  extern __shared__ unsigned char smem[];
  int* keys = (int*)smem;
  unsigned char* vals = (unsigned char*)(keys + HASH_CAPACITY);

  (void)level;

  if(idx >= survivor_count || *found_flag)
    return;

  if(threadIdx.x == 0)
  {
    Tuple5 t = survivors[idx];
    if(exact_zero_sparse(t, keys, vals))
    {
      if(atomicCAS(found_flag, 0, 1) == 0)
      {
        hit->level = t.d + t.e;
        hit->tuple = t;
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
  HitRecord* d_hit = NULL;
  HitRecord h_hit;
  int h_found = 0;
  int h_overflow = 0;
  int h_survivors = 0;
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
  cuda_check(cudaMalloc(&d_hit, sizeof(HitRecord)),
    "cudaMalloc hit");

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
      exact_level_kernel<<<h_survivors, 1, shared_bytes>>>(
        level,
        d_survivors,
        h_survivors,
        d_hit,
        d_found_flag
      );
      cuda_check(cudaGetLastError(), "launch exact_level_kernel");
      cuda_check(cudaDeviceSynchronize(), "sync exact_level_kernel");
    }

    cuda_check(cudaMemcpy(&h_found, d_found_flag, sizeof(int),
      cudaMemcpyDeviceToHost), "copy found_flag");

    printf("level=%d candidates=%llu survivors=%d found=%d\n",
      level, (unsigned long long)total, h_survivors, h_found);
    fflush(stdout);

    if(h_found)
    {
      cuda_check(cudaMemcpy(&h_hit, d_hit, sizeof(HitRecord),
        cudaMemcpyDeviceToHost), "copy hit");
      printf("HIT level=%d tuple=(%d,%d,%d,%d,%d)\n",
        h_hit.level,
        h_hit.tuple.a,
        h_hit.tuple.b,
        h_hit.tuple.c,
        h_hit.tuple.d,
        h_hit.tuple.e);
      break;
    }
  }

  cudaFree(d_survivors);
  cudaFree(d_survivor_count);
  cudaFree(d_overflow_flag);
  cudaFree(d_found_flag);
  cudaFree(d_hit);

  return 0;
}
