use anyhow::{Result, Error};
extern "C" {
    fn spacemesh_get_device_num() -> u32;
    fn spacemesh_get_max_task_num(device_idx: u32) -> u32;
    fn spacemesh_scrypt(
        device_id: u32,
        starting_index: u64,
        input: *const u32,
        task_num: u32,
        output: *mut u32,
    ) -> bool;
}

pub fn get_device_num() -> u32 {
    unsafe { spacemesh_get_device_num() }
}

pub fn get_max_task_num(device_id: u32) -> u32 {
    unsafe { spacemesh_get_max_task_num(device_id) }
}

pub fn scrypt(
    device_id: u32,
    starting_index: u64,
    input: &Vec<u32>,
    task_num: u32,
    output: &mut Vec<u8>,
) -> Result<()>{
    if unsafe {
        spacemesh_scrypt(
            device_id,
            starting_index,
            input.as_ptr(),
            task_num,
            output.as_mut_ptr() as *mut u32)
    } {
        Ok(())
    }else {
        Err(Error::msg("Failed to run scrypt process"))
    }
}
