#include <stdexcept>
#include <vector>

#include "device/kernel.cuh"
#include "spacemesh_cuda/spacemesh.h"
#include "utils.hpp"

constexpr uint32_t LOOKUP_GAP = 2;
constexpr uint32_t N = 8192;
constexpr uint32_t LOOKUP_MEM_FOR_ONE_TASK = 128 * N;
constexpr uint32_t OUTPUT_MEM_FOR_ONE_TASK = 32;

struct DeviceContext {
  size_t block_dim;
  size_t block_num;
  size_t max_task_num;
};

class Config {
public:
  DeviceContext& GetDeviceContext(size_t device_id) {
    if (device_id >= device_num_) {
      throw std::invalid_argument("device_id is invalid!");
    }

    return device_contexts_.at(device_id);
  }

  size_t GetDeviceNum() const { return device_num_; }

  static Config& GetDefault() {
    static Config conf;
    return conf;
  };

private:
  Config();
  Config(const Config&) = delete;
  Config(Config&&) = delete;
  Config& operator=(const Config&) = delete;
  Config& operator=(Config&&) = delete;

  size_t device_num_;
  std::vector<DeviceContext> device_contexts_;
};

inline Config::Config() {
  {
    int count;
    CHECK(cudaGetDeviceCount(&count));
    if (count <= 0) {
      throw "cuda device not found!";
    }
    device_num_ = count;
  }

  for (size_t di = 0; di < device_num_; ++di) {
    GPUContextSwitcher switcher(di);
    UNUSED(switcher);

    size_t free_mem;
    size_t total;
    CHECK(cudaMemGetInfo(&free_mem, &total));
    UNUSED(total);
    auto device_prop = GetDeviceProp(di);

    size_t block_num = device_prop.multiProcessorCount;
    size_t smsp_num = GetSMSPNum(device_prop.major, device_prop.minor);
    size_t block_dim = device_prop.warpSize * smsp_num;

    size_t use_mem =
        block_num * block_dim *
        (OUTPUT_MEM_FOR_ONE_TASK + LOOKUP_MEM_FOR_ONE_TASK / LOOKUP_GAP);

    while (use_mem * 2 < free_mem) {
      block_num *= 2;
      use_mem =
          block_num * block_dim *
          (OUTPUT_MEM_FOR_ONE_TASK + LOOKUP_MEM_FOR_ONE_TASK / LOOKUP_GAP);
    }
    size_t max_thread_num = block_num * block_dim;
    size_t max_task_num = block_dim * block_num;

    while (true) {
      use_mem = max_task_num * 2 * OUTPUT_MEM_FOR_ONE_TASK +
                max_thread_num * LOOKUP_MEM_FOR_ONE_TASK / LOOKUP_GAP;
      if (use_mem >= free_mem || max_task_num > 1024 * 1024 * 8) {
        break;
      } else {
        max_task_num *= 2;
      }
    }

    device_contexts_.push_back({});
    device_contexts_.back().block_dim = block_dim;
    device_contexts_.back().block_num = block_num;
    device_contexts_.back().max_task_num = max_task_num;
  }
}

uint32_t spacemesh_get_device_num() {
  return Config::GetDefault().GetDeviceNum();
}

uint32_t spacemesh_get_max_task_num(uint32_t device_idx) {
  return Config::GetDefault().GetDeviceContext(device_idx).max_task_num;
}

bool spacemesh_scrypt(uint32_t device_idx, const uint64_t starting_index,
                      const uint32_t* input, const uint32_t task_num,
                      uint32_t* output) {
  auto& ctx = Config::GetDefault().GetDeviceContext(device_idx);
  GPUContextSwitcher switcher(device_idx);
  UNUSED(switcher);

  if (task_num > ctx.max_task_num) {
    return false;
//    throw std::invalid_argument("task_num must less " +
//                                std::to_string(ctx.max_task_num));
  }

  CudaDeviceMem<uint4> d_output(task_num * 2);
  CudaDeviceMem<uint4> d_lookup(ctx.block_dim * ctx.block_num * 8 * N /
                                LOOKUP_GAP);

  uint4 input_1 = make_uint4(input[0], input[1], input[2], input[3]);
  uint4 input_2 = make_uint4(input[4], input[5], input[6], input[7]);

  scrypt_coalesce_access_v3<LOOKUP_GAP>
      <<<ctx.block_num, ctx.block_dim,
         33ULL * sizeof(uint4) / 4 * ctx.block_dim>>>(
          starting_index, task_num, input_1, input_2, d_lookup.Ptr(),
          d_output.Ptr());
  CHECK(cudaMemcpy(output, d_output.Ptr(), task_num * OUTPUT_MEM_FOR_ONE_TASK,
                   cudaMemcpyDeviceToHost));
  return true;
}
