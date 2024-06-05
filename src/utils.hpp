#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#define CHECK(call)                                                   \
  do {                                                                \
    const cudaError_t error_code = call;                              \
    if (error_code != cudaSuccess) {                                  \
      printf("CUDA Error:\n");                                        \
      printf("    File:   %s\n", __FILE__);                           \
      printf("    Line:   %d\n", __LINE__);                           \
      printf("    Error code: %d\n", error_code);                     \
      printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
      exit(1);                                                        \
    }                                                                 \
  } while (0);

class GPUContextSwitcher {
public:
  /**
   * Use CUDA GPU with index "gpu_index".
   * @param gpu_index The index of cuda capable GPU.
   */
  GPUContextSwitcher(uint32_t gpu_index) {
    CHECK(cudaGetDevice(&old_gpu_index_));
    CHECK(cudaSetDevice(gpu_index));
  }

  /**
   * Restore the GPU context that was used before creating the
   * GpuContextSwitcher if Restore() has not been called.
   */
  ~GPUContextSwitcher() { Restore(); }

  /**
   * Restore the GPU context that was used before creating the
   * GpuContextSwitcher if Restore() has not been called.
   */
  void Restore() noexcept {
    if (old_gpu_index_ != -1) {
      cudaSetDevice(old_gpu_index_);
      old_gpu_index_ = -1;
    }
  }

private:
  int old_gpu_index_;
};

static inline cudaDeviceProp GetDeviceProp(int device_id) {
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, device_id));
  return prop;
}

static inline uint32_t GetSMSPNum(int major, int minor) {
  const auto& invalid_msg = [](int major, int minor) -> std::string {
    char buf[64];
    sprintf(buf, "Invalid argument (major=%d, minor=%d) of compute capability",
            major, minor);
    return buf;
  };

  switch (major) {
    case 1: {
      switch (minor) {
        case 0:
        case 1:
        case 2:
        case 3:
          return 1;
        default:
          throw std::invalid_argument(invalid_msg(major, minor));
      }
    }
    case 2: {
      switch (minor) {
        case 0:
        case 1:
          return 1;
        default:
          throw std::invalid_argument(invalid_msg(major, minor));
      }
    }
    case 3: {
      switch (minor) {
        case 0:
        case 2:
        case 5:
        case 7:
          return 1;
        default:
          throw std::invalid_argument(invalid_msg(major, minor));
      }
    }
    case 5: {
      switch (minor) {
        case 0:
        case 2:
        case 3:
          return 4;
        default:
          throw std::invalid_argument(invalid_msg(major, minor));
      }
    }
    case 6: {
      switch (minor) {
        case 0: {
          return 2;
        }
        case 1:
        case 2: {
          return 4;
        }
        default:
          throw std::invalid_argument(invalid_msg(major, minor));
      }
    }
    case 7: {
      switch (minor) {
        case 0:
        case 2:
        case 5:
          return 4;
        default:
          throw std::invalid_argument(invalid_msg(major, minor));
      }
    }
    case 8: {
      switch (minor) {
        case 0:
        case 6:
        case 7:
        case 9:
          return 4;
        default:
          throw std::invalid_argument(invalid_msg(major, minor));
      }
    }
    case 9: {
      switch (minor) {
        case 0:
          return 4;
        default:
          throw std::invalid_argument(invalid_msg(major, minor));
      }
    }
    default:
      throw std::invalid_argument(invalid_msg(major, minor));
  }
}

/*
 * An RAII class of cuda device memory.
 * Represent a one-dimensional array stored on the device.
 * @tparam E The data type of the elements in the array.
 */
template <typename E>
class CudaDeviceMem {
public:
  /**
   * Default constructor.
   *  Doesn't allocate array on the device.
   */
  CudaDeviceMem() : CudaDeviceMem(0) {}

  /**
   * Allocates a one-dimensional array capable of accommodating element_num
   * elements on the device.
   * @param element_num
   */
  CudaDeviceMem(size_t element_num) : d_ptr_(nullptr), element_num_(0) {
    Resize(element_num);
  }

  /**
   * Allocate a device-side array of the same size as "other" and copy all the
   * data from "other" to this array.
   * @param other
   */
  CudaDeviceMem(const CudaDeviceMem& other) : CudaDeviceMem() { *this = other; }

  /**
   * Move constructor.
   * Move the device-side array stored in "other" into this object.
   * @param other
   */
  CudaDeviceMem(CudaDeviceMem&& other) noexcept : CudaDeviceMem() {
    *this = std::move(other);
  }

  /**
   * Destructor.
   */
  ~CudaDeviceMem() noexcept {
    try {
      Clear();
    } catch (...) {
      // Do nothing
    }
  }

  /**
   * Get the pointer to the device-side array.
   * @return The pointer to the device-side array.
   */
  E* Ptr() noexcept { return d_ptr_; }

  /**
   * Get the pointer to the device-side array.
   * @return The pointer to the device-side array.
   */
  const E* Ptr() const noexcept { return d_ptr_; }

  /**
   * Type conversion operator.
   * @return The pointer to the device-side array.
   */
  operator E*() noexcept { return d_ptr_; }

  /**
   * Type conversion operator.
   * @return The pointer to the device-side array.
   */
  operator const E*() const noexcept { return d_ptr_; }

  /**
   * Get the size of the array on the device side in bytes.
   * @return The size of the array on the device side, in bytes.
   */
  size_t SizeInBytes() const noexcept { return element_num_ * sizeof(E); }

  /**
   * Get the size of the array on the device side in terms of the number of
   * elements.
   * @return The size of the array on the device side in terms of the number of
   * elements.
   */
  size_t Num() const noexcept { return element_num_; }

  /**
   * Resize the array on the device side.
   * If the new array size is the same as the old array size, then simply return
   * directly.
   * @param element_num New size of the array on the device side in terms of the
   * number of elements.
   */
  void Resize(size_t element_num) {
    if (element_num_ != element_num) {
      Clear();
      if (element_num > 0) {
        CHECK(cudaMalloc(&d_ptr_, element_num * sizeof(E)));
      }
      element_num_ = element_num;
    }
  }

  /**
   * Free the array on the device side.
   */
  void Clear() {
    if (d_ptr_ != nullptr) {
      CHECK(cudaFree(d_ptr_));
      d_ptr_ = nullptr;
      element_num_ = 0;
    }
  }

  /**
   * Copy assignment operator.
   * Resize this array and copy all data from other to this array.
   * @param other
   * @return Reference to this object.
   */
  CudaDeviceMem& operator=(const CudaDeviceMem& other) {
    if (this == &other) {
      return *this;
    }

    if (other.d_ptr_ != nullptr) {
      if (element_num_ != other.element_num_) {
        Resize(other.element_num_);
      }
      CHECK(cudaMemcpy(d_ptr_, other.d_ptr_, other.SizeInBytes(),
                       cudaMemcpyDeviceToDevice));
    } else {
      Clear();
    }
    return *this;
  }

  /**
   * Move assignment operator.
   * @param other
   * @return Reference to this object.
   */
  CudaDeviceMem& operator=(CudaDeviceMem&& other) {
    if (this == &other) {
      return *this;
    }
    Clear();
    d_ptr_ = other.d_ptr_;
    other.d_ptr_ = nullptr;

    element_num_ = other.element_num_;
    other.element_num_ = 0;
    return *this;
  }

private:
  E* d_ptr_;
  size_t element_num_;
};

/**
 * An RAII class of cuda host memory.
 * Represent a one-dimensional array stored on the host side.
 * @tparam E The data type of the elements in the array.
 */
template <typename E>
class CudaHostMem {
public:
  /**
   * Default constructor.
   * Doesn't allocate array on the host side.
   */
  CudaHostMem() : CudaHostMem(0) {}

  /**
   * Allocates a one-dimensional array capable of accommodating element_num
   * elements with flag cudaHostAllocDefault on the host side.
   * @param element_num
   */
  CudaHostMem(size_t element_num)
      : d_ptr_(nullptr),
        h_ptr_(nullptr),
        element_num_(0),
        flags_(cudaHostAllocDefault) {
    Resize(element_num);
  }

  /**
   * Allocates a one-dimensional array capable of accommodating element_num
   * elements with the specified flags on the host side.
   * @param element_num
   * @param flags The flags used when allocate host-side memory. The value of
   * the flags parameter needs to be one of the flags that the cudaHostAlloc()
   * function can accept.
   */
  CudaHostMem(size_t element_num, unsigned int flags)
      : d_ptr_(nullptr),
        h_ptr_(nullptr),
        element_num_(0),
        flags_(cudaHostAllocDefault) {
    Resize(element_num, flags);
  }

  /**
   * Copy constructor.
   * Allocate a host-side array of the same size and the same flags as "other"
   * and copy all the data from "other" to this array.
   * @param other
   */
  CudaHostMem(const CudaHostMem& other) : CudaHostMem() { *this = other; }

  /**
   * Move constructor.
   * Move the host-side array stored in "other" into this object.
   * @param other
   */
  CudaHostMem(CudaHostMem&& other) noexcept : CudaHostMem() {
    *this = std::move(other);
  }

  /**
   * Destructor.
   */
  ~CudaHostMem() noexcept {
    try {
      Clear();
    } catch (...) {
      // Do nothing.
    }
  }

  /**
   * Get the host-side pointer to the array.
   * @return The host-side pointer to the array.
   */
  E* HPtr() noexcept { return h_ptr_; }

  /**
   * Get the host-side pointer to the array.
   * @return The host-side pointer to the array.
   */
  const E* HPtr() const noexcept { return h_ptr_; }

  /**
   * Get the mapped device-side pointer to the array.
   * @return The mapped device-side pointer to the array. If the array is not
   * allocated with flag cudaHostAllocMapped, nullptr is returned.
   */
  E* DPtr() noexcept { return d_ptr_; }

  /**
   * Get the mapped device-side pointer to the array.
   * @return The mapped device-side pointer to the array. If the array is not
   * allocated with flag cudaHostAllocMapped, nullptr is returned.
   */
  const E* DPtr() const noexcept { return d_ptr_; }

  /**
   * Get the size of the array on the host side in bytes.
   * @return The size of the array on the host side, in bytes.
   */
  size_t SizeInBytes() const noexcept { return element_num_ * sizeof(E); }

  /**
   * Get the size of the array on the host side in terms of the number of
   * elements.
   * @return The size of the array on the host side in terms of the number of
   * elements.
   */
  size_t Num() const noexcept { return element_num_; }

  /**
   * Resize the array on the host side.
   * If the new array size is the same as the old array size, then simply return
   * directly.
   * @param element_num New size of the array on the host side in terms of the
   * number of elements.
   */
  void Resize(size_t element_num) { Resize(element_num, flags_); }

  /**
   * Resize the array on the host side with new flags.
   * If the new array size and the new flags is the same as the old array then
   * simply return directly.
   * @param element_num New size of the array on the host side in terms of the
   * number of elements.
   * @param flags The flags used when allocate host-side memory. The value of
   * the flags parameter needs to be one of the flags that the cudaHostAlloc()
   * function can accept.
   */
  void Resize(size_t element_num, unsigned int flags) {
    if (element_num == element_num_ && flags == flags_) {
      return;
    }
    Clear();
    if (element_num > 0) {
      CHECK(cudaHostAlloc(&h_ptr_, element_num * sizeof(E), flags));
      if ((cudaHostAllocMapped & flags) != 0) {
        try {
          CHECK(cudaHostGetDevicePointer(&d_ptr_, h_ptr_, 0));
        } catch (std::exception& e) {
          Clear();
          throw e;
        }
      }
    }
    element_num_ = element_num;
    flags_ = flags;
  }

  /**
   * Free the host-side array.
   */
  void Clear() {
    if (h_ptr_ != nullptr) {
      CHECK(cudaFreeHost(h_ptr_));
    }
    h_ptr_ = d_ptr_ = nullptr;
    element_num_ = 0;
  }

  /**
   * Copy assignment operator.
   * Resize the array managed in this object with the flags used when create the
   * array managed in the other object. Then copy all data from other to the new
   * array.
   * @param other
   * @return Reference to this object.
   */
  CudaHostMem& operator=(const CudaHostMem& other) {
    if (this == &other) {
      return *this;
    }

    if (other.h_ptr_ != nullptr) {
      if (element_num_ != other.element_num_ || flags_ != other.flags_) {
        Resize(other.element_num_, other.flags_);
      }
      memcpy(h_ptr_, other.h_ptr_, other.SizeInBytes());
    } else {
      Clear();
      flags_ = other.flags_;
    }
    return *this;
  }

  /**
   * Move assignment operator.
   * @param other
   * @return Reference to this object.
   */
  CudaHostMem& operator=(CudaHostMem&& other) {
    if (this == &other) {
      return *this;
    }
    Clear();
    flags_ = other.flags_;
    other.flags_ = cudaHostAllocDefault;

    h_ptr_ = other.h_ptr_;
    other.h_ptr_ = nullptr;

    d_ptr_ = other.d_ptr_;
    other.d_ptr_ = nullptr;

    element_num_ = other.element_num_;
    other.element_num_ = 0;
    return *this;
  }

  /**
   * Obtaining the flags used when creating a one-dimensional array on the host
   * side.
   * @return The flags used when creating a one-dimensional array on the host
   * side.
   */
  unsigned int GetFlags() const noexcept { return flags_; }

private:
  E* d_ptr_;
  E* h_ptr_;
  size_t element_num_;
  unsigned int flags_;
};

#endif
