[//]: # (Image References)

[image1]: ./Figure_1.png "Rewards Plot"

# Project 1: Navigation - Report

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

### Future work
As mentioned before, due to the simplicy of the environment, even a standard DQN seems to perform adequately. However, if we anted to experement further, a double DQN, Dueling DQN or even a Rainbow approach might be a way to go.