from abc import abstractmethod
from ..data.data import OpenProtDataset

class OpenProtEval(OpenProtDataset):

    @abstractmethod
    def run(self, model): # model is OpenProtWrapper
        NotImplemented
        '''
        Probably something like:
        for batch in self:
            model.tokenize(batch)
            batch.to('cuda')
            self.run_batch(batch)
        '''

    @abstractmethod
    def run_batch(self, model, batch, device=None):
        NotImplemented
        