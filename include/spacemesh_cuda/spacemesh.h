#ifndef SPACEMESH_CUDA_SPACEMESH_H
#define SPACEMESH_CUDA_SPACEMESH_H

#include <stdint.h>

extern "C" {
/**
 * Obtain the number of devices (CUDA GPU)
 *
 * @returns A uint32
 */
uint32_t spacemesh_get_device_num();

/**
 * Obtain the maximum number of tasks of device @p device_id
 *
 * @returns A uint32
 */
uint32_t spacemesh_get_max_task_num(uint32_t device_id);

/**
 * Execution
 *
 * @param device_id The index of device
 * @param starting_index
 * @param input Input CPU memory
 * @param task_num The number of tasks
 * @param output Output CPU memory
 */
bool spacemesh_scrypt(uint32_t device_id, const uint64_t starting_index,
                      const uint32_t* input, const uint32_t task_num,
                      uint32_t* output);
}

#endif  // SPACEMESH_CUDA_SPACEMESH_H
