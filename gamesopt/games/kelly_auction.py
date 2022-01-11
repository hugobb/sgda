from gamesopt.games import Game
from typing import Optional, Tuple, List
from dataclasses import dataclass
import torch


@dataclass
class KellyAuctionConfig:
    bidding_cost: float = 100
    ressources: float = 1000
    marginal_utility_gains: Tuple[float] = (1.8, 2.0, 2.2, 2.4)

class KellyAuction(Game):
    def __init__(self, config: KellyAuctionConfig = KellyAuctionConfig()) -> None:
        N = len(config.marginal_utility_gains)
        players = [torch.zeros(1, requires_grad=True) for _ in range(N)]
        super().__init__(players, 1)

        self.Q = config.ressources
        self.Z = config.bidding_cost
        self.G = config.marginal_utility_gains

    def reset(self) -> None:
        return [torch.zeros(1, requires_grad=True) for _ in range(self.num_players)]

    def loss(self, index: Optional[int] = None) -> List[torch.Tensor]:
        loss = []
        bidding_sum = torch.cat(self.players).sum()
        for i in range(self.num_players):
            rho = self.Q*self.players[i]/(self.Z + bidding_sum)
            _loss = self.G[i]*rho - self.players[i]
            loss.append(_loss)

        return loss


