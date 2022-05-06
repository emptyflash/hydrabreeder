# hydrabreeder
Lifecoding with hydra. Hydra breeder is a genetic algorithm applied to [hydracodegenerator](https://github.com/alecominotti/hydracodegenerator).
It uses per frame entropy as a fitness function, and "breeds" two programs by randomly swapping source from each.

## Getting started
Install the dependencies with `pip3 install -r requirements.txt`. Then run `python3 main.py`.

This should open a python controlled chrome browser and begin running the genetic algorithm.
The best (most entropic) program from each generation will be displayed. 

You can tweak settings in `main.py`, primarily the number of iterations, population size,
and the number of frames to take from each program.
