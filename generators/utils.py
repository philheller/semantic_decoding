import pickle
import torch
import gc
import time
from typing import Optional, Tuple

def deep_compare(obj1, obj2, rtol: float = 1e-5):
    if type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, torch.Tensor):
        return torch.allclose(obj1, obj2, rtol=rtol)

    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_compare(obj1[key], obj2[key]) for key in obj1)

    if isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        return all(deep_compare(item1, item2) for item1, item2 in zip(obj1, obj2))

    return obj1 == obj2

def to_pickle_file(obj, path):
    with open(path, 'ab') as f:
        pickle.dump(obj, f)

def load_from_pickle_file(path):
    with open(path, 'rb') as f:
        objects = []
        while True:
                try:
                    obj = pickle.load(f)
                    objects.append(obj)
                except EOFError:
                    break
    return objects

def clean_up(*variables):
    for variable in variables:
        del variable
    torch.cuda.empty_cache()
    gc.collect()
    
def report_memory(pre: Optional[str] = None, only_max: bool = True):
    if pre is not None:
        print(pre)
    num_gpus = torch.cuda.device_count()

    total_vram_gb_list = []
    allocated_gb_list = []
    reserved_gb_list = []
    max_allocated_gb_list = []
    max_reserved_gb_list = []
    
    allocated_percentage_list = []
    reserved_percentage_list = []
    max_allocated_percentage_list = []
    max_reserved_percentage_list = []

    for device in range(num_gpus):
        props = torch.cuda.get_device_properties(device)
        total_vram = props.total_memory

        total_vram_gb = total_vram / 1024**3
        total_vram_gb_list.append(total_vram_gb)

        if not only_max:
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            max_reserved = torch.cuda.max_memory_reserved(device)

            allocated_gb = allocated / 1024**3
            reserved_gb = reserved / 1024**3
            max_reserved_gb = max_reserved / 1024**3

            allocated_percentage = (allocated / total_vram) * 100
            reserved_percentage = (reserved / total_vram) * 100
            max_reserved_percentage = (max_reserved / total_vram) * 100

            allocated_gb_list.append(allocated_gb)
            reserved_gb_list.append(reserved_gb)
            max_reserved_gb_list.append(max_reserved_gb)

            allocated_percentage_list.append(allocated_percentage)
            reserved_percentage_list.append(reserved_percentage)
            max_reserved_percentage_list.append(max_reserved_percentage)

        max_allocated = torch.cuda.max_memory_allocated(device)
        max_allocated_gb = max_allocated / 1024**3
        max_allocated_percentage = (max_allocated / total_vram) * 100

        max_allocated_gb_list.append(max_allocated_gb)
        max_allocated_percentage_list.append(max_allocated_percentage)

    # Output summary
    devices = [f"Device {i}" for i in range(num_gpus)]
    max_allocated_repr_values = [f"{max_allocated_gb:.3f}GB ({max_allocated_percentage:.3f}%)" for max_allocated_gb, max_allocated_percentage in zip(max_allocated_gb_list, max_allocated_percentage_list)]
    if not only_max:
        allocated_repr_values = [f"{allocated_gb:.3f}GB ({allocated_percentage:.3f}%)" for allocated_gb, allocated_percentage in zip(allocated_gb_list, allocated_percentage_list)]
        reserved_repr_values = [f"{reserved_gb:.3f}GB ({reserved_percentage:.3f}%)" for reserved_gb, reserved_percentage in zip(reserved_gb_list, reserved_percentage_list)]
        max_reserved_repr_values = [f"{max_reserved_gb:.3f}GB ({max_reserved_percentage:.3f}%)" for max_reserved_gb, max_reserved_percentage in zip(max_reserved_gb_list, max_reserved_percentage_list)]
    
    print("".join([f"{device:20s}" for device in devices])) 
    if not only_max:
        print("".join([f"{allocated_repr_values:20s}" for allocated_repr_values in allocated_repr_values]))
        print("".join([f"{reserved_repr_values:20s}" for reserved_repr_values in reserved_repr_values]))
    print("".join([f"{max_allocated_repr_values:20s}" for max_allocated_repr_values in max_allocated_repr_values]))
    if not only_max:
        print("".join([f"{max_reserved_repr_values:20s}" for max_reserved_repr_values in max_reserved_repr_values]))

def get_tensor_vram_usage(tensor: torch.Tensor) -> float:
    num_elements = tensor.numel()
    element_size = tensor.element_size()
    vram_usage = num_elements * element_size
    return vram_usage / (1024 ** 3)

def get_pkv_vram_usage(pkv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> float:
    vram_in_gb = 0
    for tensor_pair in pkv:
        for tensor in tensor_pair:
            vram_in_gb += get_tensor_vram_usage(tensor)


class TimeReporter:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.last_time = self.start_time
        self.on = True
        
    def reset_timer(self):
        if not self.on:
            return
        self.start_time = time.time()
        self.last_time = self.start_time

    def turn_off(self):
        self.on = False

    def turn_on(self):
        self.on = True
    
    def report_time(
        self,
        log_point: Optional[str] = None
    ) -> float:
        if not self.on:
            return 0.0
        this_time = time.time()
        delta_start = this_time - self.start_time
        delta_last = this_time - self.last_time
        self.last_time = this_time

        mins_since_start = int(delta_start // 60)
        mins_since_last = int(delta_last // 60)

        secs_since_start = int(delta_start % 60)
        secs_since_last = int(delta_last % 60)

        millis_since_start = int((delta_start % 1) * 1000)
        millis_since_last = int((delta_last % 1) * 1000)

        print(f"{mins_since_start:02d}:{secs_since_start:02d}:{millis_since_start:<3n}\t{mins_since_last:02d}:{secs_since_last:02d}:{millis_since_last:<3n}\t->{log_point}")
        return this_time