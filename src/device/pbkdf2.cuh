#ifndef DEVICE_PBKDF2_CUH
#define DEVICE_PBKDF2_CUH
#include <cuda_runtime.h>

#include "device/common.cuh"

#define SCRYPT_HASH_DIGEST_SIZE 64
#define SCRYPT_KECCAK_F 1600
#define SCRYPT_HASH_BLOCK_SIZE 72
#define SCRYPT_BLOCK_BYTES 128

typedef struct scrypt_hash_state_t {
  uint4 state4[(SCRYPT_KECCAK_F + 127) / 128];        // 8 bytes of extra
  uint4 buffer4[(SCRYPT_HASH_BLOCK_SIZE + 15) / 16];  // 8 bytes of extra
                                                      // uint leftover;
} scrypt_hash_state;

typedef struct scrypt_hmac_state_t {
  scrypt_hash_state inner;
  scrypt_hash_state outer;
} scrypt_hmac_state;

__constant__ uint64_t keccak_round_constants[24]{
  0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
  0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
  0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
  0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
  0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
  0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
  0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
  0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL};

inline __device__ void keccak_block_core(scrypt_hash_state &S) {
  uint2 t[5];
  uint2 u[5];
  uint2 v;
  uint2 w;
  uint4 *s4 = S.state4;

  for (uint i = 0; i < 24; i++) {
/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
#define CASE(D, S00, S01, S10, S11, S20, S21, S30, S31, S40, S41) \
  {                                                               \
    D.x = S00 ^ S10 ^ S20 ^ S30 ^ S40;                            \
    D.y = S01 ^ S11 ^ S21 ^ S31 ^ S41;                            \
  }
    // t[0] = s4[0].xy ^ s4[2].zw ^ s4[5].xy ^ s4[7].zw ^ s4[10].xy;
    CASE(t[0], s4[0].x, s4[0].y, s4[2].z, s4[2].w, s4[5].x, s4[5].y, s4[7].z,
         s4[7].w, s4[10].x, s4[10].y);
    // t[1] = s4[0].zw ^ s4[3].xy ^ s4[5].zw ^ s4[8].xy ^ s4[10].zw;
    CASE(t[1], s4[0].z, s4[0].w, s4[3].x, s4[3].y, s4[5].z, s4[5].w, s4[8].x,
         s4[8].y, s4[10].z, s4[10].w);
    // t[2] = s4[1].xy ^ s4[3].zw ^ s4[6].xy ^ s4[8].zw ^ s4[11].xy;
    CASE(t[2], s4[1].x, s4[1].y, s4[3].z, s4[3].w, s4[6].x, s4[6].y, s4[8].z,
         s4[8].w, s4[11].x, s4[11].y);
    // t[3] = s4[1].zw ^ s4[4].xy ^ s4[6].zw ^ s4[9].xy ^ s4[11].zw;
    CASE(t[3], s4[1].z, s4[1].w, s4[4].x, s4[4].y, s4[6].z, s4[6].w, s4[9].x,
         s4[9].y, s4[11].z, s4[11].w);
    // t[4] = s4[2].xy ^ s4[4].zw ^ s4[7].xy ^ s4[9].zw ^ s4[12].xy;
    CASE(t[4], s4[2].x, s4[2].y, s4[4].z, s4[4].w, s4[7].x, s4[7].y, s4[9].z,
         s4[9].w, s4[12].x, s4[12].y);
#undef CASE

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    u[0] = t[4] ^ rotl64<1>(t[1]);
    u[1] = t[0] ^ rotl64<1>(t[2]);
    u[2] = t[1] ^ rotl64<1>(t[3]);
    u[3] = t[2] ^ rotl64<1>(t[4]);
    u[4] = t[3] ^ rotl64<1>(t[0]);

/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
#define CASE(D0, D1, S) \
  {                     \
    D0 ^= S.x;          \
    D1 ^= S.y;          \
  }

    // s4[0].xy ^= u[0];
    CASE(s4[0].x, s4[0].y, u[0]);
    // s4[2].zw ^= u[0];
    CASE(s4[2].z, s4[2].w, u[0]);
    // s4[5].xy ^= u[0];
    CASE(s4[5].x, s4[5].y, u[0]);
    // s4[7].zw ^= u[0];
    CASE(s4[7].z, s4[7].w, u[0]);
    // s4[10].xy ^= u[0];
    CASE(s4[10].x, s4[10].y, u[0]);
    // s4[0].zw ^= u[1];
    CASE(s4[0].z, s4[0].w, u[1]);
    // s4[3].xy ^= u[1];
    CASE(s4[3].x, s4[3].y, u[1]);
    // s4[5].zw ^= u[1];
    CASE(s4[5].z, s4[5].w, u[1]);
    // s4[8].xy ^= u[1];
    CASE(s4[8].x, s4[8].y, u[1]);
    // s4[10].zw ^= u[1];
    CASE(s4[10].z, s4[10].w, u[1]);
    // s4[1].xy ^= u[2];
    CASE(s4[1].x, s4[1].y, u[2]);
    // s4[3].zw ^= u[2];
    CASE(s4[3].z, s4[3].w, u[2]);
    // s4[6].xy ^= u[2];
    CASE(s4[6].x, s4[6].y, u[2]);
    // s4[8].zw ^= u[2];
    CASE(s4[8].z, s4[8].w, u[2]);
    // s4[11].xy ^= u[2];
    CASE(s4[11].x, s4[11].y, u[2]);
    // s4[1].zw ^= u[3];
    CASE(s4[1].z, s4[1].w, u[3]);
    // s4[4].xy ^= u[3];
    CASE(s4[4].x, s4[4].y, u[3]);
    // s4[6].zw ^= u[3];
    CASE(s4[6].z, s4[6].w, u[3]);
    // s4[9].xy ^= u[3];
    CASE(s4[9].x, s4[9].y, u[3]);
    // s4[11].zw ^= u[3];
    CASE(s4[11].z, s4[11].w, u[3]);
    // s4[2].xy ^= u[4];
    CASE(s4[2].x, s4[2].y, u[4]);
    // s4[4].zw ^= u[4];
    CASE(s4[4].z, s4[4].w, u[4]);
    // s4[7].xy ^= u[4];
    CASE(s4[7].x, s4[7].y, u[4]);
    // s4[9].zw ^= u[4];
    CASE(s4[9].z, s4[9].w, u[4]);
    // s4[12].xy ^= u[4];
    CASE(s4[12].x, s4[12].y, u[4]);
#undef CASE

    /* rho pi: b[..] = rotl(a[..], ..) */
    // v = s4[0].zw;
    v = make_uint2(s4[0].z, s4[0].w);
#define CASE(D0, D1, S0, S1, M)           \
  {                                       \
    const uint2 &tmp = rotl64<M>(S0, S1); \
    D0 = tmp.x;                           \
    D1 = tmp.y;                           \
  }

    // s4[0].zw = ROTL64(s4[3].xy, 44UL);
    CASE(s4[0].z, s4[0].w, s4[3].x, s4[3].y, 44);
    // s4[3].xy = ROTL64(s4[4].zw, 20UL);
    CASE(s4[3].x, s4[3].y, s4[4].z, s4[4].w, 20);
    // s4[4].zw = ROTL64(s4[11].xy, 61UL);
    CASE(s4[4].z, s4[4].w, s4[11].x, s4[11].y, 61);
    // s4[11].xy = ROTL64(s4[7].xy, 39UL);
    CASE(s4[11].x, s4[11].y, s4[7].x, s4[7].y, 39);
    // s4[7].xy = ROTL64(s4[10].xy, 18UL);
    CASE(s4[7].x, s4[7].y, s4[10].x, s4[10].y, 18);
    // s4[10].xy = ROTL64(s4[1].xy, 62UL);
    CASE(s4[10].x, s4[10].y, s4[1].x, s4[1].y, 62);
    // s4[1].xy = ROTL64(s4[6].xy, 43UL);
    CASE(s4[1].x, s4[1].y, s4[6].x, s4[6].y, 43);
    // s4[6].xy = ROTL64(s4[6].zw, 25UL);
    CASE(s4[6].x, s4[6].y, s4[6].z, s4[6].w, 25);
    // s4[6].zw = ROTL64(s4[9].zw, 8UL);
    CASE(s4[6].z, s4[6].w, s4[9].z, s4[9].w, 8);
    // s4[9].zw = ROTL64(s4[11].zw, 56UL);
    CASE(s4[9].z, s4[9].w, s4[11].z, s4[11].w, 56);
    // s4[11].zw = ROTL64(s4[7].zw, 41UL);
    CASE(s4[11].z, s4[11].w, s4[7].z, s4[7].w, 41);
    // s4[7].zw = ROTL64(s4[2].xy, 27UL);
    CASE(s4[7].z, s4[7].w, s4[2].x, s4[2].y, 27);
    // s4[2].xy = ROTL64(s4[12].xy, 14UL);
    CASE(s4[2].x, s4[2].y, s4[12].x, s4[12].y, 14);
    // s4[12].xy = ROTL64(s4[10].zw, 2UL);
    CASE(s4[12].x, s4[12].y, s4[10].z, s4[10].w, 2);
    // s4[10].zw = ROTL64(s4[4].xy, 55UL);
    CASE(s4[10].z, s4[10].w, s4[4].x, s4[4].y, 55);
    // s4[4].xy = ROTL64(s4[8].xy, 45UL);
    CASE(s4[4].x, s4[4].y, s4[8].x, s4[8].y, 45);
    // s4[8].xy = ROTL64(s4[2].zw, 36UL);
    CASE(s4[8].x, s4[8].y, s4[2].z, s4[2].w, 36);
    // s4[2].zw = ROTL64(s4[1].zw, 28UL);
    CASE(s4[2].z, s4[2].w, s4[1].z, s4[1].w, 28);
    // s4[1].zw = ROTL64(s4[9].xy, 21UL);
    CASE(s4[1].z, s4[1].w, s4[9].x, s4[9].y, 21);
    // s4[9].xy = ROTL64(s4[8].zw, 15UL);
    CASE(s4[9].x, s4[9].y, s4[8].z, s4[8].w, 15);
    // s4[8].zw = ROTL64(s4[5].zw, 10UL);
    CASE(s4[8].z, s4[8].w, s4[5].z, s4[5].w, 10);
    // s4[5].zw = ROTL64(s4[3].zw, 6UL);
    CASE(s4[5].z, s4[5].w, s4[3].z, s4[3].w, 6);
    // s4[3].zw = ROTL64(s4[5].xy, 3UL);
    CASE(s4[3].z, s4[3].w, s4[5].x, s4[5].y, 3);
    // s4[5].xy = ROTL64(v, 1UL);
    CASE(s4[5].x, s4[5].y, v.x, v.y, 1);
#undef CASE

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
    // v = s4[0].xy;
    v = make_uint2(s4[0].x, s4[0].y);
    // w = s4[0].zw;
    w = make_uint2(s4[0].z, s4[0].w);

#define CASE(D0, D1, S00, S01, S10, S11) \
  {                                      \
    D0 ^= (~S00) & S10;                  \
    D1 ^= (~S01) & S11;                  \
  }

    // s4[0].xy ^= (~w) & s4[1].xy;
    CASE(s4[0].x, s4[0].y, w.x, w.y, s4[1].x, s4[1].y);
    // s4[0].zw ^= (~s4[1].xy) & s4[1].zw;
    CASE(s4[0].z, s4[0].w, s4[1].x, s4[1].y, s4[1].z, s4[1].w);
    // s4[1].xy ^= (~s4[1].zw) & s4[2].xy;
    CASE(s4[1].x, s4[1].y, s4[1].z, s4[1].w, s4[2].x, s4[2].y);
    // s4[1].zw ^= (~s4[2].xy) & v;
    CASE(s4[1].z, s4[1].w, s4[2].x, s4[2].y, v.x, v.y);
    // s4[2].xy ^= (~v) & w;
    CASE(s4[2].x, s4[2].y, v.x, v.y, w.x, w.y);
    // v = s4[2].zw;
    v = make_uint2(s4[2].z, s4[2].w);
    // w = s4[3].xy;
    w = make_uint2(s4[3].x, s4[3].y);
    // s4[2].zw ^= (~w) & s4[3].zw;
    CASE(s4[2].z, s4[2].w, w.x, w.y, s4[3].z, s4[3].w);
    // s4[3].xy ^= (~s4[3].zw) & s4[4].xy;
    CASE(s4[3].x, s4[3].y, s4[3].z, s4[3].w, s4[4].x, s4[4].y);
    // s4[3].zw ^= (~s4[4].xy) & s4[4].zw;
    CASE(s4[3].z, s4[3].w, s4[4].x, s4[4].y, s4[4].z, s4[4].w);
    // s4[4].xy ^= (~s4[4].zw) & v;
    CASE(s4[4].x, s4[4].y, s4[4].z, s4[4].w, v.x, v.y);
    // s4[4].zw ^= (~v) & w;
    CASE(s4[4].z, s4[4].w, v.x, v.y, w.x, w.y);
    // v = s4[5].xy;
    v = make_uint2(s4[5].x, s4[5].y);
    // w = s4[5].zw;
    w = make_uint2(s4[5].z, s4[5].w);
    // s4[5].xy ^= (~w) & s4[6].xy;
    CASE(s4[5].x, s4[5].y, w.x, w.y, s4[6].x, s4[6].y);
    // s4[5].zw ^= (~s4[6].xy) & s4[6].zw;
    CASE(s4[5].z, s4[5].w, s4[6].x, s4[6].y, s4[6].z, s4[6].w);
    // s4[6].xy ^= (~s4[6].zw) & s4[7].xy;
    CASE(s4[6].x, s4[6].y, s4[6].z, s4[6].w, s4[7].x, s4[7].y);
    // s4[6].zw ^= (~s4[7].xy) & v;
    CASE(s4[6].z, s4[6].w, s4[7].x, s4[7].y, v.x, v.y);
    // s4[7].xy ^= (~v) & w;
    CASE(s4[7].x, s4[7].y, v.x, v.y, w.x, w.y);
    // v = s4[7].zw;
    v = make_uint2(s4[7].z, s4[7].w);
    // w = s4[8].xy;
    w = make_uint2(s4[8].x, s4[8].y);
    // s4[7].zw ^= (~w) & s4[8].zw;
    CASE(s4[7].z, s4[7].w, w.x, w.y, s4[8].z, s4[8].w);
    // s4[8].xy ^= (~s4[8].zw) & s4[9].xy;
    CASE(s4[8].x, s4[8].y, s4[8].z, s4[8].w, s4[9].x, s4[9].y);
    // s4[8].zw ^= (~s4[9].xy) & s4[9].zw;
    CASE(s4[8].z, s4[8].w, s4[9].x, s4[9].y, s4[9].z, s4[9].w);
    // s4[9].xy ^= (~s4[9].zw) & v;
    CASE(s4[9].x, s4[9].y, s4[9].z, s4[9].w, v.x, v.y);
    // s4[9].zw ^= (~v) & w;
    CASE(s4[9].z, s4[9].w, v.x, v.y, w.x, w.y);
    // v = s4[10].xy;
    v = make_uint2(s4[10].x, s4[10].y);
    // w = s4[10].zw;
    w = make_uint2(s4[10].z, s4[10].w);
    // s4[10].xy ^= (~w) & s4[11].xy;
    CASE(s4[10].x, s4[10].y, w.x, w.y, s4[11].x, s4[11].y);
    // s4[10].zw ^= (~s4[11].xy) & s4[11].zw;
    CASE(s4[10].z, s4[10].w, s4[11].x, s4[11].y, s4[11].z, s4[11].w);
    // s4[11].xy ^= (~s4[11].zw) & s4[12].xy;
    CASE(s4[11].x, s4[11].y, s4[11].z, s4[11].w, s4[12].x, s4[12].y);
    // s4[11].zw ^= (~s4[12].xy) & v;
    CASE(s4[11].z, s4[11].w, s4[12].x, s4[12].y, v.x, v.y);
    // s4[12].xy ^= (~v) & w;
    CASE(s4[12].x, s4[12].y, v.x, v.y, w.x, w.y);
#undef CASE

    /* iota: a[0,0] ^= round constant */
    // s4[0].xy ^= as_uint2(keccak_round_constants[i]);
    s4[0].x ^= uint32_t(keccak_round_constants[i]);
    s4[0].y ^= uint32_t(keccak_round_constants[i] >> UINT32_NUM_BITS);
  }
}

inline __device__ void keccak_block(scrypt_hash_state &S, const uint4 *in4) {
  uint4 *s4 = S.state4;
  uint i;

  /* absorb input */
  for (i = 0; i < 4; i++) {
    s4[i].x ^= in4[i].x;
    s4[i].y ^= in4[i].y;
    s4[i].z ^= in4[i].z;
    s4[i].w ^= in4[i].w;
  }

  s4[4].x ^= in4[4].x;
  s4[4].y ^= in4[4].y;

  keccak_block_core(S);
}

inline __device__ void keccak_block_zero(scrypt_hash_state &S,
                                         const uint4 *in4) {
  uint4 *s4 = S.state4;
  uint i;

  /* absorb input */
  for (i = 0; i < 4; i++) {
    s4[i] = in4[i];
  }
  // s4[4].xyzw = (uint4)(in4[4].xy, 0, 0);
  s4[4] = make_uint4(in4[4].x, in4[4].y, 0, 0);

  for (i = 5; i < 12; i++) {
    // s4[i] = ZERO;
    s4[i] = make_zero<uint4>();
  }
  // s4[12].xy = ZERO_UINT2;
  s4[12].x = 0;
  s4[12].y = 0;

  keccak_block_core(S);
}

inline __device__ void scrypt_hash_update_72(scrypt_hash_state &S,
                                             const uint4 *in4) {
  /* handle the current data */
  keccak_block_zero(S, in4);
}

inline __device__ void scrypt_hash_update_80(scrypt_hash_state &S,
                                             const uint4 *in4) {
  const uchar1 *in = (const uchar1 *)in4;
  // uint i;

  /* handle the current data */
  keccak_block(S, in4);
  in += SCRYPT_HASH_BLOCK_SIZE;

  /* handle leftover data */
  // S->leftover = 2;

  {
    const uint2 *in2 = (const uint2 *)in;

    // S->buffer4[0].xy = int2[0].xy;
    S.buffer4[0].x = in2[0].x;
    S.buffer4[0].y = in2[0].y;
  }
}

inline __device__ void scrypt_hash_update_128(scrypt_hash_state &S,
                                              const uint4 *in4) {
  const uchar1 *in = (const uchar1 *)in4;
  // uint i;

  /* handle the current data */
  keccak_block(S, in4);
  in += SCRYPT_HASH_BLOCK_SIZE;

  /* handle leftover data */
  // S->leftover = 14;

  {
    const uint2 *in2 = (const uint2 *)in;

    for (uint i = 0; i < 3; i++) {
      S.buffer4[i] = make_uint4(in2[2 * i].x, in2[2 * i].y, in2[2 * i + 1].x,
                                in2[2 * i + 1].y);
    }
    // S->buffer4[3].xy = int2[6].xy;
    S.buffer4[3].x = in2[6].x;
    S.buffer4[3].y = in2[6].y;
  }
}

inline __device__ void scrypt_hash_update_4_after_72(scrypt_hash_state &S,
                                                     uint in) {
  S.buffer4[0] = make_uint4(in, 0x01, 0, 0);
}

inline __device__ void scrypt_hash_update_4_after_80(scrypt_hash_state &S,
                                                     uint in) {
  // assume that leftover = 2
  /* handle the previous data */
  // S->buffer4[0].zw = (uint2)(in, 0x01);
  S.buffer4[0].z = in;
  S.buffer4[0].w = 0x01;
  // S->leftover += 1;
}

inline __device__ void scrypt_hash_update_4_after_128(scrypt_hash_state &S,
                                                      uint in) {
  // leftover = 14
  /* handle the previous data */
  // S->buffer4[3].zw = (uint2)(in, 0x01);
  S.buffer4[3].z = in;
  S.buffer4[3].w = 0x01;
  // S->leftover += 1;
}

inline __device__ void scrypt_hash_update_64(scrypt_hash_state &S,
                                             const uint4 *in4) {
  /* handle leftover data */
  // S->leftover = 16;
  for (uint32_t i = 0; i < 4; i++) {
    S.buffer4[i] = in4[i];
  }
}

inline __device__ void scrypt_hash_finish_80_after_64(scrypt_hash_state &S,
                                                      uint4 *hash4) {
  // assume that leftover = 16
  // S->buffer4[4].xy = (uint2)(0x01, 0x80000000);
  S.buffer4[4].x = 0x01;
  S.buffer4[4].y = 0x80000000;

  keccak_block(S, S.buffer4);

  for (uint i = 0; i < 4; i++) {
    hash4[i] = S.state4[i];
  }
}

inline __device__ void scrypt_hash_finish_80_after_80_4(scrypt_hash_state &S,
                                                        uint4 *hash4) {
  // assume that leftover = 3
  // S->buffer4[0].w = 0x01; // done already in scrypt_hash_update_4_after_80
  for (uint i = 1; i < 4; i++) {
    S.buffer4[i] = make_zero<uint4>();
  }
  // S->buffer4[4].xy = (uint2)(0, 0x80000000);
  S.buffer4[4].x = 0;
  S.buffer4[4].y = 0x80000000;

  keccak_block(S, S.buffer4);

  for (uint i = 0; i < 4; i++) {
    hash4[i] = S.state4[i];
  }
}

inline __device__ void scrypt_hash_finish_80_after_128_4(scrypt_hash_state &S,
                                                         uint4 *hash4) {
  // leftover = 15
  // S->buffer4[3].w = 0x01; // done already in scrypt_hash_update_4_after_128
  // S->buffer4[4].xy = (uint2)(0, 0x80000000);
  S.buffer4[4].x = 0;
  S.buffer4[4].y = 0x80000000;

  keccak_block(S, S.buffer4);

  for (uint i = 0; i < 4; i++) {
    hash4[i] = S.state4[i];
  }
}

inline __device__ void scrypt_hash_72(uint4 *hash4, const uint4 *m) {
  for (uint i = 0; i < 4; i++) {
    hash4[i] = m[i];
  }
  // hash4[4].xy = m[4].xy;
  hash4[4].x = m[4].x;
  hash4[4].y = m[4].y;
}

inline __device__ void scrypt_hash_80(uint4 *hash4, const uint4 *m) {
  const uchar1 *in = (const uchar1 *)m;
  scrypt_hash_state st;

  /* handle the current data */
  keccak_block_zero(st, m);
  in += SCRYPT_HASH_BLOCK_SIZE;

  {
    const uint2 *in2 = (const uint2 *)in;
    // st.buffer4[0].xyzw = (uint4)(in2[0].xy, 0x01, 0);
    st.buffer4[0] = make_uint4(in2[0].x, in2[0].y, 0x01, 0);
  }

  for (uint i = 1; i < 4; i++) {
    st.buffer4[i] = make_zero<uint4>();
  }
  // st.buffer4[4].xyzw = (uint4)(0, 0x80000000, 0, 0);
  st.buffer4[4] = make_uint4(0, 0x80000000, 0, 0);

  keccak_block(st, st.buffer4);

  for (uint i = 0; i < 4; i++) {
    hash4[i] = st.state4[i];
  }
}

/* hmac */
constexpr uint KEY_0X36 = 0x36363636;
constexpr uint KEY_0X36_XOR_0X5C = 0x6A6A6A6A;

inline __device__ void scrypt_hmac_init(scrypt_hmac_state &st,
                                        const uint4 *key) {
  uint4 pad4[SCRYPT_HASH_BLOCK_SIZE / 16 + 1];

  scrypt_hash_72(pad4, key);

  /* inner = (key ^ 0x36) */
  /* h(inner || ...) */
  for (uint i = 0; i < 4; i++) {
    pad4[i].x ^= KEY_0X36;
    pad4[i].y ^= KEY_0X36;
    pad4[i].z ^= KEY_0X36;
    pad4[i].w ^= KEY_0X36;
  }
  // pad4[4].xy ^= KEY_0X36_2;
  pad4[4].x ^= KEY_0X36;
  pad4[4].y ^= KEY_0X36;

  scrypt_hash_update_72(st.inner, pad4);

  /* outer = (key ^ 0x5c) */
  /* h(outer || ...) */
  for (uint i = 0; i < 4; i++) {
    pad4[i].x ^= KEY_0X36_XOR_0X5C;
    pad4[i].y ^= KEY_0X36_XOR_0X5C;
    pad4[i].z ^= KEY_0X36_XOR_0X5C;
    pad4[i].w ^= KEY_0X36_XOR_0X5C;
  }
  // pad4[4].xy ^= KEY_0X36_XOR_0X5C_2;
  pad4[4].x ^= KEY_0X36_XOR_0X5C;
  pad4[4].y ^= KEY_0X36_XOR_0X5C;

  scrypt_hash_update_72(st.outer, pad4);
}

inline __device__ void scrypt_hmac_update_80(scrypt_hmac_state &st,
                                             const uint4 *m) {
  /* h(inner || m...) */
  scrypt_hash_update_80(st.inner, m);
}

inline __device__ void scrypt_hmac_update_72(scrypt_hmac_state &st,
                                             const uint4 *m) {
  /* h(inner || m...) */
  scrypt_hash_update_72(st.inner, m);
}

inline __device__ void scrypt_hmac_update_128(scrypt_hmac_state &st,
                                              const uint4 *m) {
  /* h(inner || m...) */
  scrypt_hash_update_128(st.inner, m);
}

inline __device__ void scrypt_hmac_update_4_after_72(scrypt_hmac_state &st,
                                                     uint m) {
  /* h(inner || m...) */
  scrypt_hash_update_4_after_72(st.inner, m);
}

inline __device__ void scrypt_hmac_update_4_after_80(scrypt_hmac_state &st,
                                                     uint m) {
  /* h(inner || m...) */
  scrypt_hash_update_4_after_80(st.inner, m);
}

inline __device__ void scrypt_hmac_update_4_after_128(scrypt_hmac_state &st,
                                                      uint m) {
  /* h(inner || m...) */
  scrypt_hash_update_4_after_128(st.inner, m);
}

inline __device__ void scrypt_hmac_finish_128B(scrypt_hmac_state &st,
                                               uint4 *mac) {
  /* h(inner || m) */
  uint4 innerhash[4];
  scrypt_hash_finish_80_after_80_4(st.inner, innerhash);

  /* h(outer || h(inner || m)) */
  scrypt_hash_update_64(st.outer, innerhash);
  scrypt_hash_finish_80_after_64(st.outer, mac);
}

inline __device__ void scrypt_hmac_finish_32B(scrypt_hmac_state &st,
                                              uint4 *mac) {
  /* h(inner || m) */
  uint4 innerhash[4];
  scrypt_hash_finish_80_after_128_4(st.inner, innerhash);

  /* h(outer || h(inner || m)) */
  scrypt_hash_update_64(st.outer, innerhash);
  scrypt_hash_finish_80_after_64(st.outer, mac);
}

inline __device__ void scrypt_copy_hmac_state_128B(
  scrypt_hmac_state &dest, const scrypt_hmac_state &src) {
  for (uint i = 0; i < 12; i++) {
    dest.inner.state4[i] = src.inner.state4[i];
  }
  // dest->inner.state4[12].xy = src->inner.state4[12].xy;
  dest.inner.state4[12].x = src.inner.state4[12].x;
  dest.inner.state4[12].y = src.inner.state4[12].y;

  // dest->inner.buffer4[0].xy = src->inner.buffer4[0].xy;
  dest.inner.buffer4[0].x = src.inner.buffer4[0].x;
  dest.inner.buffer4[0].y = src.inner.buffer4[0].y;

  for (uint i = 0; i < 12; i++) {
    dest.outer.state4[i] = src.outer.state4[i];
  }
  // dest->outer.state4[12].xy = src->outer.state4[12].xy;
  dest.outer.state4[12].x = src.outer.state4[12].x;
  dest.outer.state4[12].y = src.outer.state4[12].y;
}

constexpr uint be1 = 0x01000000;
constexpr uint be2 = 0x02000000;

inline __device__ void scrypt_pbkdf2_128B(const uint4 *password, uint4 *out4) {
  scrypt_hmac_state hmac_pw, work;
  uint4 ti4[4];

  /* bytes must be <= (0xffffffff - (SCRYPT_HASH_DIGEST_SIZE - 1)), which they
   * will always be under scrypt */

  /* hmac(password, ...) */
  scrypt_hmac_init(hmac_pw, password);

  /* hmac(password, salt...) */
  // Skip salt
  // scrypt_hmac_update_80(&hmac_pw, salt);

  /* U1 = hmac(password, salt || be(i)) */
  /* U32TO8_BE(be, i); */
  // work = hmac_pw;
  scrypt_copy_hmac_state_128B(work, hmac_pw);
  scrypt_hmac_update_4_after_72(work, be1);
  scrypt_hmac_finish_128B(work, ti4);

  for (uint i = 0; i < 4; i++) {
    out4[i] = ti4[i];
  }

  /* U1 = hmac(password, salt || be(i)) */
  /* U32TO8_BE(be, i); */
  // work = hmac_pw;
  scrypt_hmac_update_4_after_72(hmac_pw, be2);
  scrypt_hmac_finish_128B(hmac_pw, ti4);

  for (uint i = 0; i < 4; i++) {
    out4[i + 4] = ti4[i];
  }
}

inline __device__ void scrypt_pbkdf2_32B(const uint4 *password,
                                         const uint4 *salt,
                                         uint4 *__restrict__ out4) {
  scrypt_hmac_state hmac_pw;
  uint4 ti4[4];

  /* bytes must be <= (0xffffffff - (SCRYPT_HASH_DIGEST_SIZE - 1)), which they
   * will always be under scrypt */

  /* hmac(password, ...) */
  scrypt_hmac_init(hmac_pw, password);

  /* hmac(password, salt...) */
  scrypt_hmac_update_128(hmac_pw, salt);

  /* U1 = hmac(password, salt || be(i)) */
  /* U32TO8_BE(be, i); */
  scrypt_hmac_update_4_after_128(hmac_pw, be1);
  scrypt_hmac_finish_32B(hmac_pw, ti4);

  for (uint i = 0; i < 2; i++) {
    out4[i] = ti4[i];
  }
}
#endif
