# Code for [Stochastic Gradient Descent-Ascent: Unified Theory and New Efficient Methods](https://arxiv.org/abs/2202.07262)

## Requirements
- `torch`

## Installation
`pip install -e .`

## Testing
`python -m unittest discover -s tests`

## Notebooks
To reproduce the results of the paper, run the notebooks in the `notebooks` folder.

## Documentation
### Games
If you want to test the different optimizers on your own game, create a class that inherits from `Game`. 
You should implement either the `loss` or `operator` method, and the `sample` method.

### Optimizers
If you want to test your own optimizer on the existing games, create a class that inherits from `Optimizer`.
You should implement the `step` method that update the parameters of the players.

### Distributed
QSGDA, DIANA-SGDA and VR-DIANA-SGDA are implemented using `torch.distributed`. 

## Folder structure
```
- gamesopt:
  - games:
    - quadratic_games.py  # code for the definition of quadratic games.
  - optimizer:
    - sgda.py   # code for SVRGDA and L-SVRGDA
    - distributed.py  # code for QSGDA, DIANA-SGDA and VR-DIANA-SGDA
  - train.py  # To run the experiments
  - train_distributed #to run the distributed experiments.
```
