# hydrabreeder
Lifecoding with hydra. Hydra breeder is a genetic algorithm applied to [hydracodegenerator](https://github.com/alecominotti/hydracodegenerator).
It uses per frame entropy as a fitness function, and "breeds" two programs by randomly swapping sources from each.

## Getting started
Install the dependencies with `pip3 install -r requirements.txt`. Then run `python3 main.py`.

This should open a python controlled chrome browser and begin running the genetic algorithm.
The best (most entropic) program from each generation will be displayed. 

You can tweak settings in `main.py`, primarily the number of iterations, population size,
and the number of frames to take from each program.

For livecoding performances, you can tweak the program to look better and fit the music more while
the next generation is running. Here's an example of the kind of results this produced at the [Chaos2 Algorave](https://youtu.be/QDE03-YZON4) in NYC.

## Future work
This thing can produce some pretty cool stuff. However, much of it can be a little too chaotic for most livecoding purposes.

I think this is mostly because the ultimate goal of optimizing for entropy is to just produce pure noise.

It may make sense to come up with a better fitness function, either by manually extracting features
(entropy, change between frames, color count, etc.) of what makes a hydra sketch "look good", or by training a GAN on
stills from existing hydra sketches and using the discriminator to determine fitness.
