import pickle
import torch
from pynvml import *

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

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()