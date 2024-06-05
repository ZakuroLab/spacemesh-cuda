#ifndef DEVICE_COMMON_CUH
#define DEVICE_COMMON_CUH

#include <cuda_runtime.h>

#include <cstdint>

#define UINT32_NUM_BITS sizeof(uint32_t) * 8
#define UINT64_NUM_BITS sizeof(uint64_t) * 8

#define UNUSED(X) (void)(X)

template <uint32_t d>
__device__ uint32_t rotl32(uint32_t x) {
  return (x << d) | (x >> (UINT32_NUM_BITS - d));
}

template <uint32_t d>
__device__ uint2 rotl64(uint32_t x, uint32_t y) {
  uint64_t c = uint64_t(x) | (uint64_t(y) << UINT32_NUM_BITS);
  c = (c << d) | (c >> (UINT64_NUM_BITS - d));
  return make_uint2(uint32_t(c), c >> UINT32_NUM_BITS);
}
template <uint32_t d>
__device__ uint2 rotl64(uint2 v) {
  return rotl64<d>(v.x, v.y);
}

inline __device__ uint2 operator^(const uint2 &t0, const uint2 &t1) {
  return {t0.x ^ t1.x, t0.y ^ t1.y};
}
inline __device__ uint2 &operator^=(uint2 &t0, const uint2 &t1) {
  t0.x ^= t1.x;
  t0.y ^= t1.y;
  return t0;
}

inline __device__ uint4 operator^(const uint4 &t0, const uint4 &t1) {
  return {t0.x ^ t1.x, t0.y ^ t1.y, t0.z ^ t1.z, t0.w ^ t1.w};
}

inline __device__ uint4 &operator^=(uint4 &t0, const uint4 &t1) {
  t0.x ^= t1.x;
  t0.y ^= t1.y;
  t0.z ^= t1.z;
  t0.w ^= t1.w;
  return t0;
}

inline __device__ uint4 &operator+=(uint4 &t0, const uint4 &t1) {
  t0.x += t1.x;
  t0.y += t1.y;
  t0.z += t1.z;
  t0.w += t1.w;
  return t0;
}

template <typename T>
inline __device__ T make_zero();

template <>
inline __device__ uint4 make_zero<uint4>() {
  return {0, 0, 0, 0};
}
template <>
inline __device__ uint2 make_zero<uint2>() {
  return {0, 0};
}

#endif
