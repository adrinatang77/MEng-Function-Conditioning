from .task import Task
import numpy as np


class SequenceGeneration(Task):
    def prep_data(self, data):
        data["_seq_noise"] = 1
        data["_seq_supervise"] = 1
        data["_struct_noise"] = 0
        data["_struct_supervise"] = 0
        return data
