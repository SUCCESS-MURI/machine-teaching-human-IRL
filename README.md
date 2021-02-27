# Machine Teaching for Human Inverse Reinforcement Learning

This repository primarily consists of 

* Methods of a) selecting demonstrations that effectively summarize the agent's policy to a human (augments a [BEC summary](https://arxiv.org/pdf/1805.07687.pdf) of the agent's policy with scaffolding and visual optimization), and b) requesting demonstrations of what the human believes an agent would do in specific environments. See main.py
* Implementations of various MDP testbeds using David Abel's [simple_rl framework](https://github.com/david-abel/simple_rl). 

models/* contains the training and testing demonstrations used in the user study, which were created using Python 3.5.

Required packages include [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/), [pypoman](https://github.com/stephane-caron/pypoman) to perform computational geometry with polytopes (i.e. BEC regions, see below), and [pygame](http://www.pygame.org/news) if you want to visualize some MDPs.

This repository contains raw code that has not gone extensive cleanup. If you have any questions, please contact the primary author at ml5@andrew.cmu.edu.

