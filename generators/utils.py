import pickle
import torch
import gc
from typing import Optional

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
    
def report_memory(pre: Optional[str] = None):
    if pre is not None:
        print(pre)
    print(f"Memory allocated:\t{torch.cuda.memory_allocated()//1024**2:6f} MB")
    print(f"Max memory allocated:\t{torch.cuda.max_memory_allocated()//1024**2:6f} MB")
    print(f"Memory reserved:\t{torch.cuda.memory_reserved()//1024**2:6f} MB")
    print(f"Max memory reserved:\t{torch.cuda.max_memory_reserved()//1024**2:6f} MB")