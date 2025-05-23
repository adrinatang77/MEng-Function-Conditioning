from abc import abstractmethod
from ..data.data import OpenProtDataset
from ..model.wrapper import OpenProtWrapper


class OpenProtEval(OpenProtDataset):

    @abstractmethod
    def run(self, model: OpenProtWrapper):
        NotImplemented
        """
        Probably something like:
        for batch in self:
            model.tokenize(batch)
            batch.to('cuda')
            self.run_batch(batch)
        """

    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):
        pass

    @abstractmethod
    def run_batch(
        self,
        model: OpenProtWrapper,
        batch: dict,
        noisy_batch: dict = None,
        device=None
    ):
        NotImplemented
