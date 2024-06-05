#include <cuda_runtime.h>

#include <chrono>

#include "device/kernel.cuh"
#include "gtest/gtest.h"
#include "utils.hpp"

TEST(Spacemesh, PerfCoalesceOrg) {
  constexpr uint32_t LOOKUP_GAP = 2;
  constexpr uint32_t task_num_per_thread = 128;

  uint4 in0{2839345266U, 42009750U, 875455879U, 2217211394U};
  uint4 in1{3438177526U, 2734532412U, 2819254414U, 1408356118U};

  const uint32_t N = 8 * 1024;
  uint64_t starting_index = 0;

  auto device_prop = GetDeviceProp(0);
  uint32_t smsp_num = GetSMSPNum(device_prop.major, device_prop.minor);
  uint32_t core_num =
      device_prop.warpSize * smsp_num * device_prop.multiProcessorCount;

  uint32_t thread_num = core_num * LOOKUP_GAP;
  uint32_t task_num = thread_num * task_num_per_thread;

  CudaDeviceMem<uint4> d_out(task_num * 2);
  CudaDeviceMem<uint32_t> d_lookup(N / LOOKUP_GAP * 32 * thread_num);
  CudaHostMem<uint4> h_out(task_num * 2);

  constexpr size_t block_dim = 256;
  size_t block_num = thread_num / block_dim;
  size_t iter = 1;

  auto st = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iter; ++i) {
    scrypt_org<LOOKUP_GAP><<<block_num, block_dim>>>(
        starting_index, task_num, in0, in1,
        reinterpret_cast<uint4 *>(d_lookup.Ptr()), d_out.Ptr());
  }
  CHECK(cudaMemcpy(h_out.HPtr(), d_out.Ptr(), d_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));
  auto ed = std::chrono::steady_clock::now();

  double d =
      std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
  std::cout << "[org] block_num: " << block_num << ", block_dim: " << block_dim
            << ", time: " << d / iter << "ms, throughput: "
            << task_num * iter / d * 1000 * 16.0 / 1024 / 1024 << "MB/s\n";
}

TEST(Spacemesh, PerfCoalesceV1) {
  constexpr uint32_t LOOKUP_GAP = 2;
  constexpr uint32_t task_num_per_thread = 128;

  uint4 in0{2839345266U, 42009750U, 875455879U, 2217211394U};
  uint4 in1{3438177526U, 2734532412U, 2819254414U, 1408356118U};

  const uint32_t N = 8 * 1024;
  uint64_t starting_index = 0;

  auto device_prop = GetDeviceProp(0);
  uint32_t smsp_num = GetSMSPNum(device_prop.major, device_prop.minor);
  uint32_t core_num =
      device_prop.warpSize * smsp_num * device_prop.multiProcessorCount;

  uint32_t thread_num = core_num * LOOKUP_GAP;
  uint32_t task_num = thread_num * task_num_per_thread;

  CudaDeviceMem<uint4> d_out(task_num * 2);
  CudaDeviceMem<uint32_t> d_lookup(N / LOOKUP_GAP * 32 * thread_num);
  CudaHostMem<uint4> h_out(task_num * 2);

  constexpr size_t block_dim = 256;
  size_t block_num = thread_num / block_dim;
  size_t iter = 1;

  auto st = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iter; ++i) {
    scrypt_coalesce_access_v1<LOOKUP_GAP>
        <<<block_num, block_dim, 33 * sizeof(uint32_t) * block_dim>>>(
            starting_index, task_num, in0, in1,
            reinterpret_cast<uint32_t *>(d_lookup.Ptr()), d_out.Ptr());
  }
  CHECK(cudaMemcpy(h_out.HPtr(), d_out.Ptr(), d_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));
  auto ed = std::chrono::steady_clock::now();

  double d =
      std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
  std::cout << "[v1] block_num: " << block_num << ", block_dim: " << block_dim
            << ", time: " << d / iter << "ms, throughput: "
            << task_num * iter / d * 1000 * 16.0 / 1024 / 1024 << "MB/s\n";
}

TEST(Spacemesh, PerfCoalesceV2) {
  constexpr uint32_t LOOKUP_GAP = 2;
  constexpr uint32_t task_num_per_thread = 512;

  uint4 in0{2839345266U, 42009750U, 875455879U, 2217211394U};
  uint4 in1{3438177526U, 2734532412U, 2819254414U, 1408356118U};

  const uint32_t N = 8 * 1024;
  uint64_t starting_index = 0;

  auto device_prop = GetDeviceProp(0);
  uint32_t smsp_num = GetSMSPNum(device_prop.major, device_prop.minor);
  uint32_t core_num =
      device_prop.warpSize * smsp_num * device_prop.multiProcessorCount;

  uint32_t thread_num = core_num * LOOKUP_GAP;
  uint32_t task_num = thread_num * task_num_per_thread;

  CudaDeviceMem<uint4> d_out(task_num * 2);
  CudaDeviceMem<uint32_t> d_lookup(N / LOOKUP_GAP * 32 * thread_num);
  CudaHostMem<uint4> h_out(task_num * 2);

  constexpr size_t block_dim = 256;
  size_t block_num = thread_num / block_dim;
  size_t iter = 1;

  auto st = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iter; ++i) {
    scrypt_coalesce_access_v2<LOOKUP_GAP>
        <<<block_num, block_dim, 33 * sizeof(uint64_t) * block_dim / 2>>>(
            starting_index, task_num, in0, in1,
            reinterpret_cast<uint64_t *>(d_lookup.Ptr()), d_out.Ptr());
  }
  CHECK(cudaMemcpy(h_out.HPtr(), d_out.Ptr(), d_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));
  auto ed = std::chrono::steady_clock::now();

  double d =
      std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
  std::cout << "[v2] block_num: " << block_num << ", block_dim: " << block_dim
            << ", time: " << d / iter << "ms, throughput: "
            << task_num * iter / d * 1000 * 16.0 / 1024 / 1024 << "MB/s\n";
}

TEST(Spacemesh, PerfCoalesceV3) {
  constexpr uint32_t LOOKUP_GAP = 2;
  constexpr uint32_t task_num_per_thread = 512;

  uint4 in0{2839345266U, 42009750U, 875455879U, 2217211394U};
  uint4 in1{3438177526U, 2734532412U, 2819254414U, 1408356118U};

  const uint32_t N = 8 * 1024;
  uint64_t starting_index = 0;

  auto device_prop = GetDeviceProp(0);
  uint32_t smsp_num = GetSMSPNum(device_prop.major, device_prop.minor);
  uint32_t core_num =
      device_prop.warpSize * smsp_num * device_prop.multiProcessorCount;

  uint32_t thread_num = core_num * LOOKUP_GAP;
  uint32_t task_num = thread_num * task_num_per_thread;

  CudaDeviceMem<uint4> d_out(task_num * 2);
  CudaDeviceMem<uint32_t> d_lookup(N / LOOKUP_GAP * 32 * thread_num);
  CudaHostMem<uint4> h_out(task_num * 2);

  constexpr size_t block_dim = 256;
  size_t block_num = thread_num / block_dim;
  size_t iter = 1;

  auto st = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iter; ++i) {
    scrypt_coalesce_access_v3<LOOKUP_GAP>
        <<<block_num, block_dim, 33 * sizeof(uint4) * block_dim / 4>>>(
            starting_index, task_num, in0, in1,
            reinterpret_cast<uint4 *>(d_lookup.Ptr()), d_out.Ptr());
  }
  CHECK(cudaMemcpy(h_out.HPtr(), d_out.Ptr(), d_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));
  auto ed = std::chrono::steady_clock::now();

  double d =
      std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
  std::cout << "[v3] block_num: " << block_num << ", block_dim: " << block_dim
            << ", time: " << d / iter << "ms, throughput: "
            << task_num * iter / d * 1000 * 16.0 / 1024 / 1024 << "MB/s\n";
}
