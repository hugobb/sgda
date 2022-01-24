from .base import Game

class ConvexConcaveGame(Game):
    def __init__(self, players: List[torch.Tensor], num_samples: int) -> None:
        super().__init__(players, num_samples)
