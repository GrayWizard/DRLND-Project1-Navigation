[//]: # (Image References)

[image1]: ./Figure_1.png "Rewards Plot"

# Project 1: Navigation

### Introduction

The goal of the project is to train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Solution

#### Approach

My initial approach was to try a vanilla DQN learning algorithm described in Lectures 1-7 of Lesson 2. The training code (adapted from the lectures is contained in `train_agent.py` and the agent and network model in `dqn_agent.py` and `model.py` accordingly.

#### Neural Network Architecture
The network is a simple 4-layer fully connected neural network (with ReLu activation) with 37 units in the input layer, 64 units in each of the hidden layers and 4 units in the output layer.

#### Hyperparameters
The initial approach used the following hyperparameters:

DQN:
- n_episodes (int): maximum number of training episodes: *2000*
- max_t (int): maximum number of timesteps per episode: *1000*
- eps_start (float): starting value of epsilon, for epsilon-greedy action selection: *1.0*
- eps_end (float): minimum value of epsilon: *0.01*
- eps_decay (float): multiplicative factor (per episode) for decreasing epsilon: *0.995*

Agent:
- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE minibatch size: *64*
- GAMMA: discount factor: *0.99*
- TAU: for soft update of target parameters: *1e-3*
- LR: learning rate: *5e-4*
- UPDATE_EVERY: how often to update the network: *4*

#### Results

To my surprise the basic vanilla DQN with initial hyperparameters performed adequately and trained the agent to solve the environment with average score of +13 in 540 episodes.

See the rewards plot below:
![Rewards Plot][image1]

### Instructions

The project requires the installation of the environment provided by Udacity; see the detailed instructions [here](https://classroom.udacity.com/nanodegrees/nd893/parts/6b0c03a7-6667-4fcf-a9ed-dd41a2f76485/modules/4eeb16ab-5ac5-47bf-974d-12784e9730d7/lessons/69bd42c6-b70e-4866-9764-9bfa8c03cdea/concepts/319dc918-bd2c-4d3b-80a5-063bb5f1905a). The following Python 3.5 libraries are required as well (if not provided by the Udacity DRLND environment): `unityagents`,`numpy`,`torch`,`matplotlib`.

After the enviromnent is set up and activated, run `python train_agent.py` to train the agent and `python demo_trained_agent.py` to see how the trained agent performs.

### Future work
As mentioned before, due to the simplicy of the environment, even a standard DQN seems to perform adequately. However, if we anted to experement further, a double DQN, Dueling DQN or even a Rainbow approach might be a way to go.