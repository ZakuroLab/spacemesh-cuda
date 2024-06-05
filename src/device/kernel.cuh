#ifndef DEVICE_KERNEL_CUH
#define DEVICE_KERNEL_CUH

#include "device/common.cuh"
#include "device/pbkdf2.cuh"
#include "device/romix.cuh"

/**
 * Mainly modified thread mapping relationships, LOOKUP_GAP, Block, etc.,
 * Throughtput: 4.16MB/s
 */
template <uint32_t LOOKUP_GAP = 1>
__global__ void scrypt_org(const uint64_t starting_index,
                           const uint32_t num_tasks, const uint4 input_1,
                           const uint4 input_2,
                           uint4 *const __restrict__ padcache,
                           uint4 *const __restrict__ output) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t tnum = gridDim.x * blockDim.x;

  uint4 password[5];
  uint4 X[8];
  for (uint32_t t = tid; t < num_tasks; t += tnum) {
    const uint64_t index = starting_index + t;

    password[0] = input_1;
    password[1] = input_2;
    password[2].x = uint32_t(index & 0xFFFFFFFF);
    password[2].y = uint32_t((index >> 32) & 0xFFFFFFFF);
    password[2].z = 0;
    password[2].w = 0;
    password[3] = make_zero<uint4>();
    password[4] = make_zero<uint4>();

    scrypt_pbkdf2_128B(password, X);
    scrypt_ROMix_org<LOOKUP_GAP>(X, padcache, tnum, tid);
    scrypt_pbkdf2_32B(password, X, &output[t * 2]);
  }
}

/**
 * Optimize memory access and collaborate with Warp to complete data copying
 * Throughtput: 5.34MB/s
 */
template <uint32_t LOOKUP_GAP = 1>
__global__ void scrypt_coalesce_access_v1(const uint64_t starting_index,
                                          const uint32_t num_tasks,
                                          const uint4 input_1,
                                          const uint4 input_2,
                                          uint32_t *const __restrict__ padcache,
                                          uint4 *const __restrict__ output) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t tnum = gridDim.x * blockDim.x;

  uint4 password[5];
  uint4 X[8];
  for (uint32_t t = tid; t < num_tasks; t += tnum) {
    const uint64_t index = starting_index + t;

    password[0] = input_1;
    password[1] = input_2;
    password[2].x = uint32_t(index & 0xFFFFFFFF);
    password[2].y = uint32_t((index >> 32) & 0xFFFFFFFF);
    password[2].z = 0;
    password[2].w = 0;
    password[3] = make_zero<uint4>();
    password[4] = make_zero<uint4>();

    scrypt_pbkdf2_128B(password, X);
    scrypt_ROMix_coalesce_access_v1<LOOKUP_GAP>(X, padcache, tnum, tid);
    scrypt_pbkdf2_32B(password, X, &output[t * 2]);
  }
}

/**
 * Optimize memory access, with 16 threads forming a sub warp, each thread
 * performing read and write operations in units of 8 bytes (uint64_t)
 * Throughtput: 5.94MB/s
 */
template <uint32_t LOOKUP_GAP = 1>
__global__ void scrypt_coalesce_access_v2(const uint64_t starting_index,
                                          const uint32_t num_tasks,
                                          const uint4 input_1,
                                          const uint4 input_2,
                                          uint64_t *const __restrict__ padcache,
                                          uint4 *const __restrict__ output) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t tnum = gridDim.x * blockDim.x;

  uint4 password[5];
  uint4 X[8];
  for (uint32_t t = tid; t < num_tasks; t += tnum) {
    const uint64_t index = starting_index + t;

    password[0] = input_1;
    password[1] = input_2;
    password[2].x = uint32_t(index & 0xFFFFFFFF);
    password[2].y = uint32_t((index >> 32) & 0xFFFFFFFF);
    password[2].z = 0;
    password[2].w = 0;
    password[3] = make_zero<uint4>();
    password[4] = make_zero<uint4>();

    scrypt_pbkdf2_128B(password, X);
    scrypt_ROMix_coalesce_access_v2<LOOKUP_GAP>(X, padcache, tnum, tid);
    scrypt_pbkdf2_32B(password, X, &output[t * 2]);
  }
}

/**
 * Optimize memory access, with 8 threads forming a sub warp, each thread
 * performing read and write operations in units of 8 bytes (uint4) + global
 * memory -> shared memory asynchronous reads
 * Throughtput: 6.03MB/s
 */
template <uint32_t LOOKUP_GAP = 1>
__global__ void scrypt_coalesce_access_v3(const uint64_t starting_index,
                                          const uint32_t num_tasks,
                                          const uint4 input_1,
                                          const uint4 input_2,
                                          uint4 *const __restrict__ padcache,
                                          uint4 *const __restrict__ output) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t tnum = gridDim.x * blockDim.x;

  uint4 password[5];
  uint4 X[8];
  for (uint32_t t = tid; t < num_tasks; t += tnum) {
    const uint64_t index = starting_index + t;

    password[0] = input_1;
    password[1] = input_2;
    password[2].x = uint32_t(index & 0xFFFFFFFF);
    password[2].y = uint32_t((index >> 32) & 0xFFFFFFFF);
    password[2].z = 0;
    password[2].w = 0;
    password[3] = make_zero<uint4>();
    password[4] = make_zero<uint4>();

    scrypt_pbkdf2_128B(password, X);
    scrypt_ROMix_coalesce_access_v3<LOOKUP_GAP>(X, padcache, tnum, tid);
    scrypt_pbkdf2_32B(password, X, &output[t * 2]);
  }
}

#endif
