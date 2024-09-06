# Machine Teaching for Human Inverse Reinforcement Learning

This repository contains code for methods described in the following paper: 

- Michael S. Lee, Henny Admoni, Reid Simmons **Machine Teaching for Human Inverse Reinforcement Learning**, Frontiers in Robotics and AI, 2021,

which introduces a method for teaching robot decision making to humans through demonstrations of the robot's key decisions in a domain. 

We model humans as inverse reinforcement leaners to calculate information gain for possible demonstrations that could be shown, but further bias the demonstration selection for human understanding using various principles from education and cognitive science such as scaffolding, simplicity, and pattern recognition. 

The [paper](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.693050/full) and [user study](https://github.com/SUCCESS-MURI/psiturk-machine-teaching) used to validate our methods, as well as follow-on work, are available at https://symikelee.github.io/

## Running the code

Run `main.py` to select a) demonstrations that effectively summarize the robot's policy to a human (augments a [BEC summary](https://arxiv.org/pdf/1805.07687.pdf) of the robot's policy with scaffolding and visual optimization), and b) tests that ask the human to demonstrate what they believe the robot will do in new situations. 

`models/*` contains the training and testing demonstrations used in the user study, which were created using Python 3.5.

Required packages include [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/), [pypoman](https://github.com/stephane-caron/pypoman) to perform computational geometry with polytopes (i.e. BEC regions, see below), and [pygame](http://www.pygame.org/news) if you want to visualize some MDPs.

This repository also contains implementations of various MDP testbeds built using David Abel's [simple_rl framework](https://github.com/david-abel/simple_rl). 
