from .task import Task
import numpy as np


class StructurePrediction(Task):
    def prep_data(self, data):
        data["_seq_noise"] = 0
        data["_seq_supervise"] = 0
        data["_struct_noise"] = 1
        data["_struct_supervise"] = 1
        return data
