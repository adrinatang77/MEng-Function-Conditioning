import importlib
import contextlib
import numpy as np


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def autoimport(name):
    module, name_ = name.rsplit(".", 1)
    return getattr(importlib.import_module(module), name_)
