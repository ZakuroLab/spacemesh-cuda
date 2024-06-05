#ifndef DEVICE_ROMIX_CUH
#define DEVICE_ROMIX_CUH

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "device/common.cuh"

constexpr uint4 MASK_2{1, 2, 3, 0}, MASK_3{2, 3, 0, 1}, MASK_4{3, 0, 1, 2},
    ROTATE_16{16, 16, 16, 16}, ROTATE_12{12, 12, 12, 12}, ROTATE_8{8, 8, 8, 8},
    ROTATE_7{7, 7, 7, 7};

inline __device__ void chacha_core(uint4 *__restrict__ state) {
  uint4 x[4];
  uint4 t;

  x[0] = state[0];
  x[1] = state[1];
  x[2] = state[2];
  x[3] = state[3];

#define CASE(D_x, D_y, D_z, D_w, S_x, S_y, S_z, S_w, R) \
  D_x = rotl32<R.x>(S_x);                               \
  D_y = rotl32<R.y>(S_y);                               \
  D_z = rotl32<R.z>(S_z);                               \
  D_w = rotl32<R.w>(S_w);

  for (uint32_t rounds = 0; rounds < 4; rounds++) {
    x[0] += x[1];
    t = x[3] ^ x[0];
    // x[3] = ROTL32(t, ROTATE_16);
    CASE(x[3].x, x[3].y, x[3].z, x[3].w, t.x, t.y, t.z, t.w, ROTATE_16);
    x[2] += x[3];
    t = x[1] ^ x[2];
    // x[1] = ROTL32(t, ROTATE_12);
    CASE(x[1].x, x[1].y, x[1].z, x[1].w, t.x, t.y, t.z, t.w, ROTATE_12);
    x[0] += x[1];
    t = x[3] ^ x[0];
    // x[3] = ROTL32(t, ROTATE_8);
    CASE(x[3].x, x[3].y, x[3].z, x[3].w, t.x, t.y, t.z, t.w, ROTATE_8);
    x[2] += x[3];
    t = x[1] ^ x[2];
    // x[1] = ROTL32(t, ROTATE_7);
    CASE(x[1].x, x[1].y, x[1].z, x[1].w, t.x, t.y, t.z, t.w, ROTATE_7);

    // x[1] = shuffle(x[1], MASK_2);
    // x[2] = shuffle(x[2], MASK_3);
    // x[3] = shuffle(x[3], MASK_4);

    // x[0] += x[1].yzwx;
    x[0] += make_uint4(x[1].y, x[1].z, x[1].w, x[1].x);
    // t = x[3].wxyz ^ x[0];
    t = make_uint4(x[3].w, x[3].x, x[3].y, x[3].z) ^ x[0];
    // x[3].wxyz = ROTL32(t, ROTATE_16);
    CASE(x[3].w, x[3].x, x[3].y, x[3].z, t.x, t.y, t.z, t.w, ROTATE_16);
    // x[2].zwxy += x[3].wxyz;
    x[2] += make_uint4(x[3].y, x[3].z, x[3].w, x[3].x);
    // t = x[1].yzwx ^ x[2].zwxy;
    t = make_uint4(x[1].y, x[1].z, x[1].w, x[1].x) ^
        make_uint4(x[2].z, x[2].w, x[2].x, x[2].y);
    // x[1].yzwx = ROTL32(t, ROTATE_12);
    CASE(x[1].y, x[1].z, x[1].w, x[1].x, t.x, t.y, t.z, t.w, ROTATE_12);
    // x[0] += x[1].yzwx;
    x[0] += make_uint4(x[1].y, x[1].z, x[1].w, x[1].x);
    // t = x[3].wxyz ^ x[0];
    t = make_uint4(x[3].w, x[3].x, x[3].y, x[3].z) ^ x[0];
    // x[3].wxyz = ROTL32(t, ROTATE_8);
    CASE(x[3].w, x[3].x, x[3].y, x[3].z, t.x, t.y, t.z, t.w, ROTATE_8);
    // x[2].zwxy += x[3].wxyz;
    x[2] += make_uint4(x[3].y, x[3].z, x[3].w, x[3].x);
    // t = x[1].yzwx ^ x[2].zwxy;
    t = make_uint4(x[1].y, x[1].z, x[1].w, x[1].x) ^
        make_uint4(x[2].z, x[2].w, x[2].x, x[2].y);
    // x[1].yzwx = ROTL32(t, ROTATE_7);
    CASE(x[1].y, x[1].z, x[1].w, x[1].x, t.x, t.y, t.z, t.w, ROTATE_7);

    // x[1] = shuffle(x[1], MASK_4);
    // x[2] = shuffle(x[2], MASK_3);
    // x[3] = shuffle(x[3], MASK_2);
  }
#undef CASE

  state[0] += x[0];
  state[1] += x[1];
  state[2] += x[2];
  state[3] += x[3];
}

inline __device__ void scrypt_ChunkMix_inplace_Bxor_local(
    uint4 *__restrict__ B /*[chunkWords]*/,
    uint4 *__restrict__ Bxor /*[chunkWords]*/) {
  /* 1: X = B_{2r - 1} */

  /* 2: for i = 0 to 2r - 1 do */
  /* 3: X = H(X ^ B_i) */
  B[0] ^= B[4] ^ Bxor[4] ^ Bxor[0];
  B[1] ^= B[5] ^ Bxor[5] ^ Bxor[1];
  B[2] ^= B[6] ^ Bxor[6] ^ Bxor[2];
  B[3] ^= B[7] ^ Bxor[7] ^ Bxor[3];

  /* SCRYPT_MIX_FN */
  chacha_core(B);

  /* 4: Y_i = X */
  /* 6: B'[0..r-1] = Y_even */
  /* 6: B'[r..2r-1] = Y_odd */

  /* 3: X = H(X ^ B_i) */
  B[4] ^= B[0] ^ Bxor[4];
  B[5] ^= B[1] ^ Bxor[5];
  B[6] ^= B[2] ^ Bxor[6];
  B[7] ^= B[3] ^ Bxor[7];

  /* SCRYPT_MIX_FN */
  chacha_core(B + 4);

  /* 4: Y_i = X */
  /* 6: B'[0..r-1] = Y_even */
  /* 6: B'[r..2r-1] = Y_odd */
}

inline __device__ void scrypt_ChunkMix_inplace_local(
    uint4 *__restrict__ B /*[chunkWords]*/) {
  /* 1: X = B_{2r - 1} */

  /* 2: for i = 0 to 2r - 1 do */
  /* 3: X = H(X ^ B_i) */
  B[0] ^= B[4];
  B[1] ^= B[5];
  B[2] ^= B[6];
  B[3] ^= B[7];

  /* SCRYPT_MIX_FN */
  chacha_core(B);

  /* 4: Y_i = X */
  /* 6: B'[0..r-1] = Y_even */
  /* 6: B'[r..2r-1] = Y_odd */

  /* 3: X = H(X ^ B_i) */
  B[4] ^= B[0];
  B[5] ^= B[1];
  B[6] ^= B[2];
  B[7] ^= B[3];

  /* SCRYPT_MIX_FN */
  chacha_core(B + 4);

  /* 4: Y_i = X */
  /* 6: B'[0..r-1] = Y_even */
  /* 6: B'[r..2r-1] = Y_odd */
}

#define Coord(x, y, z) x + y *(x##SIZE) + z *(y##SIZE) * (x##SIZE)
#define CO Coord(z, x, y)

template <uint32_t LOOKUP_GAP>
inline __device__ void scrypt_ROMix_org(uint4 *__restrict__ X,
                                        uint4 *__restrict__ lookup,
                                        uint32_t tnum, uint32_t tid) {
  constexpr uint32_t N = 8 * 1024;
  const uint32_t zSIZE = 8;
  const uint32_t ySIZE = (N / LOOKUP_GAP + (N % LOOKUP_GAP > 0));
  UNUSED(ySIZE);
  const uint32_t xSIZE = tnum;
  const uint32_t x = tid % xSIZE;
  uint32_t i = 0, j = 0, y = 0, z = 0;
  uint4 W[8];

  /* 1: X = B */
  /* implicit */

  /* 2: for i = 0 to N - 1 do */
  for (y = 0; y < N / LOOKUP_GAP; y++) {
    /* 3: V_i = X */
    for (z = 0; z < zSIZE; z++) {
      lookup[CO] = X[z];
    }

    for (j = 0; j < LOOKUP_GAP; j++) {
      /* 4: X = H(X) */
      scrypt_ChunkMix_inplace_local(X);
    }
  }

  /* 6: for i = 0 to N - 1 do */
  for (i = 0; i < N; i++) {
    /* 7: j = Integerify(X) % N */
    j = X[4].x & (N - 1);
    y = j / LOOKUP_GAP;

    for (z = 0; z < zSIZE; z++) {
      W[z] = lookup[CO];
    }

    if constexpr (LOOKUP_GAP == 2) {
      if (j & 1) {
        scrypt_ChunkMix_inplace_local(W);
      }
    }

    if constexpr (LOOKUP_GAP > 2) {
      uint c = j % LOOKUP_GAP;
      for (uint k = 0; k < c; k++) {
        scrypt_ChunkMix_inplace_local(W);
      }
    }

    /* 8: X = H(X ^ V_j) */
    scrypt_ChunkMix_inplace_Bxor_local(X, W);
  }

  /* 10: B' = X */
  /* implicit */
}

template <uint32_t LOOKUP_GAP>
inline __device__ void scrypt_ROMix_coalesce_access_v1(
    uint4 *__restrict__ X, uint32_t *__restrict__ lookup, uint32_t tnum,
    uint32_t tid) {
  extern __shared__ uint32_t smem[];
  const uint32_t row_length = warpSize + 1;
  uint32_t warp_work_space_size = row_length * warpSize;
  uint32_t warp_id = threadIdx.x / warpSize;
  uint32_t lane_id = threadIdx.x % warpSize;
  const uint32_t smem_base_offset = warp_work_space_size * warp_id;

  constexpr uint32_t N = 8 * 1024;
  const uint32_t zSIZE = 32;
  const uint32_t ySIZE = (N / LOOKUP_GAP + (N % LOOKUP_GAP > 0));
  UNUSED(ySIZE);
  const uint32_t xSIZE = tnum;
  const uint32_t x = tid % xSIZE;
  uint4 W[8];

  /* 1: X = B */
  /* implicit */

  /* 2: for i = 0 to N - 1 do */
  for (uint32_t y = 0; y < N / LOOKUP_GAP; y++) {
    uint32_t smem_offset = smem_base_offset + row_length * lane_id;
    for (uint32_t z = 0; z < zSIZE / 4; z++) {
      smem[smem_offset++] = X[z].x;
      smem[smem_offset++] = X[z].y;
      smem[smem_offset++] = X[z].z;
      smem[smem_offset++] = X[z].w;
    }
    __syncwarp();
    uint32_t offset = x * zSIZE + y * xSIZE * zSIZE;
    for (uint32_t k = 0; k < 128 / sizeof(uint32_t); k++) {
      uint32_t cur_offset = __shfl_sync(0xffffffff, offset, k);
      lookup[cur_offset + lane_id] =
          smem[smem_base_offset + k * row_length + lane_id];
    }
    __syncwarp();

    for (uint32_t j = 0; j < LOOKUP_GAP; j++) {
      /* 4: X = H(X) */
      scrypt_ChunkMix_inplace_local(X);
    }
  }

  /* 6: for i = 0 to N - 1 do */
  for (uint32_t i = 0; i < N; i++) {
    /* 7: j = Integerify(X) % N */
    uint32_t j = X[4].x & (N - 1);
    uint32_t y = j / LOOKUP_GAP;

    uint32_t offset = x * zSIZE + y * xSIZE * zSIZE;
    for (uint32_t k = 0; k < 128 / sizeof(uint32_t); k++) {
      uint32_t cur_offset = __shfl_sync(0xffffffff, offset, k);
      smem[smem_base_offset + k * row_length + lane_id] =
          lookup[cur_offset + lane_id];
    }
    uint32_t smem_offset = smem_base_offset + row_length * lane_id;
    for (uint32_t z = 0; z < zSIZE / 4; z++) {
      W[z].x = smem[smem_offset++];
      W[z].y = smem[smem_offset++];
      W[z].z = smem[smem_offset++];
      W[z].w = smem[smem_offset++];
    }
    __syncwarp();

    if constexpr (LOOKUP_GAP == 2) {
      if (j & 1) {
        scrypt_ChunkMix_inplace_local(W);
      }
    }

    if constexpr (LOOKUP_GAP > 2) {
      uint c = j % LOOKUP_GAP;
      for (uint k = 0; k < c; k++) {
        scrypt_ChunkMix_inplace_local(W);
      }
    }

    /* 8: X = H(X ^ V_j) */
    scrypt_ChunkMix_inplace_Bxor_local(X, W);
  }

  /* 10: B' = X */
  /* implicit */
}

template <uint32_t LOOKUP_GAP>
__device__ void scrypt_ROMix_coalesce_access_v2(uint4 *__restrict__ X,
                                                uint64_t *__restrict__ lookup,
                                                uint32_t tnum, uint32_t tid) {
  extern __shared__ uint64_t smem_v2[];
  const uint32_t sub_warp_size = 16;
  const uint32_t row_length = warpSize + 1;
  uint32_t warp_work_space_size = row_length * sub_warp_size;
  uint32_t warp_id = threadIdx.x / warpSize;
  uint32_t sub_warp_id = threadIdx.x / sub_warp_size % 2;
  uint32_t lane_id = threadIdx.x % warpSize;
  uint32_t sub_lane_id = threadIdx.x % sub_warp_size;
  const uint32_t smem_base_offset = warp_work_space_size * warp_id;

  constexpr uint32_t N = 8 * 1024;
  const uint32_t zSIZE = 16;
  const uint32_t ySIZE = (N / LOOKUP_GAP + (N % LOOKUP_GAP > 0));
  UNUSED(ySIZE);
  const uint32_t xSIZE = tnum;
  const uint32_t x = tid % xSIZE;
  uint4 W[8];

  /* 1: X = B */
  /* implicit */

  /* 2: for i = 0 to N - 1 do */
  for (uint32_t y = 0; y < N / LOOKUP_GAP; y++) {
    uint32_t smem_offset = smem_base_offset + row_length * sub_lane_id +
                           sub_warp_id * sub_warp_size;
    for (uint32_t z = 0; z < 128 / sizeof(uint4); z++) {
      smem_v2[smem_offset++] = (uint64_t)X[z].x | ((uint64_t)X[z].y << 32);
      smem_v2[smem_offset++] = (uint64_t)X[z].z | ((uint64_t)X[z].w << 32);
    }
    __syncwarp();
    uint32_t offset = x * zSIZE + y * xSIZE * zSIZE;
    for (uint32_t k = 0; k < 128 / sizeof(uint64_t); k++) {
      uint32_t cur_offset = __shfl_sync(0xffffffff, offset, k, 16);
      lookup[cur_offset + sub_lane_id] =
          smem_v2[smem_base_offset + k * row_length + lane_id];
    }
    __syncwarp();

    for (uint32_t j = 0; j < LOOKUP_GAP; j++) {
      /* 4: X = H(X) */
      scrypt_ChunkMix_inplace_local(X);
    }
  }

  /* 6: for i = 0 to N - 1 do */
  for (uint32_t i = 0; i < N; i++) {
    /* 7: j = Integerify(X) % N */
    uint32_t j = X[4].x & (N - 1);
    uint32_t y = j / LOOKUP_GAP;

    uint32_t offset = x * zSIZE + y * xSIZE * zSIZE;
    uint64_t tmp[16];
    for (uint32_t k = 0; k < 128 / sizeof(uint64_t); k++) {
      uint32_t cur_offset = __shfl_sync(0xffffffff, offset, k, 16);
      tmp[k] = lookup[cur_offset + sub_lane_id];
    }
    __syncwarp();
    for (uint32_t k = 0; k < 128 / sizeof(uint64_t); k++) {
      smem_v2[smem_base_offset + k * row_length + lane_id] = tmp[k];
    }
    __syncwarp();

    uint32_t smem_offset = smem_base_offset + row_length * sub_lane_id +
                           sub_warp_id * sub_warp_size;
    for (uint32_t z = 0; z < 128 / sizeof(uint4); z++) {
      auto t = smem_v2[smem_offset++];
      W[z].x = t;
      W[z].y = t >> 32;
      t = smem_v2[smem_offset++];
      W[z].z = t;
      W[z].w = t >> 32;
    }
    __syncwarp();

    if constexpr (LOOKUP_GAP == 2) {
      if (j & 1) {
        scrypt_ChunkMix_inplace_local(W);
      }
    }

    if constexpr (LOOKUP_GAP > 2) {
      uint c = j % LOOKUP_GAP;
      for (uint k = 0; k < c; k++) {
        scrypt_ChunkMix_inplace_local(W);
      }
    }

    /* 8: X = H(X ^ V_j) */
    scrypt_ChunkMix_inplace_Bxor_local(X, W);
  }

  /* 10: B' = X */
  /* implicit */
}

template <uint32_t LOOKUP_GAP>
__device__ void scrypt_ROMix_coalesce_access_v3(uint4 *__restrict__ X,
                                                uint4 *__restrict__ lookup,
                                                uint32_t tnum, uint32_t tid) {
  namespace cg = cooperative_groups;
  extern __shared__ uint4 smem_v4[];
  const uint32_t sub_warp_size = 8;
  const uint32_t row_length = warpSize + 1;
  auto tb = cg::this_thread_block();
  auto tile = cg::tiled_partition<sub_warp_size>(tb);  // wrapSize=32

  uint32_t warp_work_space_size = row_length * sub_warp_size;
  uint32_t warp_id = threadIdx.x / warpSize;
  uint32_t lane_id = threadIdx.x % warpSize;
  uint32_t sub_warp_id = lane_id / sub_warp_size;
  uint32_t sub_lane_id = threadIdx.x % sub_warp_size;
  const uint32_t smem_base_offset = warp_work_space_size * warp_id;

  constexpr uint32_t N = 8 * 1024;
  const uint32_t zSIZE = 8;
  const uint32_t ySIZE = (N / LOOKUP_GAP + (N % LOOKUP_GAP > 0));
  UNUSED(ySIZE);
  const uint32_t xSIZE = tnum;
  const uint32_t x = tid % xSIZE;
  uint4 W[8];

  /* 1: X = B */
  /* implicit */

  /* 2: for i = 0 to N - 1 do */
  for (uint32_t y = 0; y < N / LOOKUP_GAP; y++) {
    uint32_t smem_offset = smem_base_offset + row_length * sub_lane_id +
                           sub_warp_id * sub_warp_size;
    for (uint32_t z = 0; z < 128 / sizeof(uint4); z++) {
      smem_v4[smem_offset++] = X[z];
    }
    __syncwarp();
    uint32_t offset = x * zSIZE + y * xSIZE * zSIZE;
    for (uint32_t k = 0; k < 128 / sizeof(uint4); k++) {
      uint32_t cur_offset = __shfl_sync(0xffffffff, offset, k, 8);
      lookup[cur_offset + sub_lane_id] =
          smem_v4[smem_base_offset + k * row_length + lane_id];
    }
    __syncwarp();

    for (uint32_t j = 0; j < LOOKUP_GAP; j++) {
      /* 4: X = H(X) */
      scrypt_ChunkMix_inplace_local(X);
    }
  }

  /* 6: for i = 0 to N - 1 do */
  for (uint32_t i = 0; i < N; i++) {
    /* 7: j = Integerify(X) % N */
    uint32_t j = X[4].x & (N - 1);
    uint32_t y = j / LOOKUP_GAP;

    uint32_t offset = x * zSIZE + y * xSIZE * zSIZE;
    // uint4 tmp[8];
    for (uint32_t k = 0; k < 128 / sizeof(uint4); k++) {
      uint32_t cur_offset = __shfl_sync(0xffffffff, offset, k, 8);
      // tmp[k] = lookup[cur_offset + sub_lane_id];
      cg::memcpy_async(tile,
                       &smem_v4[smem_base_offset + k * row_length +
                                sub_warp_id * sub_warp_size],
                       sub_warp_size, &lookup[cur_offset], sub_warp_size);
    }
    // __syncwarp();
    // for (uint32_t k = 0; k < 128 / sizeof(uint4); k++) {
    //   smem_v4[smem_base_offset + k * row_length + lane_id] = tmp[k];
    // }
    // __syncwarp();
    cg::sync(tile);
    cg::wait(tile);
    uint32_t smem_offset = smem_base_offset + row_length * sub_lane_id +
                           sub_warp_id * sub_warp_size;
    for (uint32_t z = 0; z < 128 / sizeof(uint4); z++) {
      W[z] = smem_v4[smem_offset++];
    }
    __syncwarp();

    if constexpr (LOOKUP_GAP == 2) {
      if (j & 1) {
        scrypt_ChunkMix_inplace_local(W);
      }
    }

    if constexpr (LOOKUP_GAP > 2) {
      uint c = j % LOOKUP_GAP;
      for (uint k = 0; k < c; k++) {
        scrypt_ChunkMix_inplace_local(W);
      }
    }
    /* 8: X = H(X ^ V_j) */
    scrypt_ChunkMix_inplace_Bxor_local(X, W);
  }

  /* 10: B' = X */
  /* implicit */
}

#endif
