# Report
This is the report of my experiments in training my multi-DDPG agent. 

## Learning algorithm
I followed the implementation in this [paper](https://arxiv.org/pdf/1706.02275.pdf)
The two players in this environment are identical, so I used the same DDPG actor for both the players.
The actor takes alternatively the observed state from each of the player and predicts their actions.
The critic takes all the states observed by both agents and all the actions, as explained in the above paper.
The experience replay is shared between the two players.


The models for actor and critic are a set of 2 fully connected layers of 256 nodes:


The agent makes use of an experience replay buffer from where it randomly samples previous `(state, action, reward, next_state)` tuples to build the batch for training.

### Parameters
The actor and critic network are made of two hidden fully connected layers of 256 and 128 nodes respectively.

Other parameters for the DDPG agent are:
- Replay buffer size: `100000`
- Minibatch size: `128`
- Discount factor gamma: `0.99`
- Tau for soft update of target parameters: `0.2`
- Learning rate of the actor: `0.0001`
- Learning rate of the critic: `0.001`
- Weight decay: `0`

After trying some different values, I decided to keep them as set by the DDPG agent explained in the lesson.

I did not introduce a gradient clipping like suggested in the project introduction because it did not yield any improvement during training.

## Results
By using the same actor to train both players I managed to get results relatively faster than expected.
The environment was solved in 140 episodes, although the multi-agent setup shows instability and the DPPG agent sometimes takes longer or does not learn anything at all.
Below is the plot of scores in training episodes.

![](test.png)

Scores of 3 runs using the fully trained model: `[39.17849912429229, 38.85649913148954, 39.13799912519753]`

## Future work
Plans for improving my results include:
- implement suggestions from the project introduction, like being less aggressive in updates per step
- improve the actor and critic networks by adding batch normalisation or changing the architecture
- try out different algorithms like PPO
 
