# Code for "Stochastic Algorithms for VI problems:  Unified Theory and New Efficient Methods"

## Requirements
- `torch`

## Installation
`pip install -e .`

## Testing
`python -m unittest discover -s tests`

## Documentation
### Games
If you want to test the different optimizers on your own game, create a class that inherits from `Game`. 
You should implement either the `loss` or `operator` method, and the `sample` method.

### If you want to use your own 
If you want to test your own optimizer on the existing games, create a class that inherits from `Optimizer`.
You should implement the `step` method that update the parameters of the players.

