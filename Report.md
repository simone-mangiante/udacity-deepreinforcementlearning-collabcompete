# Report
This is the report of my experiments in training my multi-DDPG agent. 

## Learning algorithm
I followed the implementation in this [paper](https://arxiv.org/pdf/1706.02275.pdf)
The two players in this environment are identical, so I used the same DDPG actor for both the players.
The actor takes alternatively the observed state from each of the player and predicts their actions.
The critic takes all the states observed by both agents and all the actions, as explained in the above paper.
The experience replay is shared between the two players.

The models for actor and critic are a set of 2 fully connected layers of 256 nodes:
- the actor takes the the state size as input and outputs the best likely action (for a single agent)
- the critic processes states and actions from all the agents to output their Q-value

The agent makes use of an experience replay buffer from where it randomly samples previous `(state, action, reward, next_state)` tuples to build the batch for training.

### Parameters
Other parameters for the DDPG agent are:
- Replay buffer size: `100000`
- Minibatch size: `128`
- Discount factor gamma: `0.99`
- Tau for soft update of target parameters: `0.2`
- Learning rate of the actor: `0.0001`
- Learning rate of the critic: `0.001`
- Weight decay: `0`

I used an implementation of Ornstein-Uhlenbeck Noise from a previous project (derived from [here](https://github.com/floodsung/DDPG/blob/master/ou_noise.py)) with decay to favour exploitation over exploration as the training progresses.
I also set gradient clipping to 1.

The size of the hidden layers in the actor and critic models seemed very important because the agents could not learn with smaller fully connected layers.
No weight decay and a relatively high `tau` (compared to the paper) played a significant role in making the multi-agent setup learn properly.

## Results
By using the same actor to train both players I managed to get results relatively faster than expected.
The environment was solved in 521 episodes, although the multi-agent setup shows instability and the DPPG agent sometimes takes longer or does not learn anything at all.
Below is the plot of scores in training episodes.

![](test.png)

Scores of 3 runs using the fully trained model: `[1.4000000208616257, 0.9000000134110451, 0.10000000149011612]`

## Future work
Plans for improving my results include:
- implement batch normalisation as suggested by other papers
- run multiple learning loops for the same sample experience: it may improve the overall score and make learning even faster
- try out different algorithms like PPO
 
