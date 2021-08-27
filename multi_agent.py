from model import Actor, Critic
from torch.optim import Adam
import torch
import numpy as np
from buffer import ReplayBuffer

# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 0.2  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
SEED = 42

class DDPGAgent:
    def __init__(self, state_size, action_size, random_seed=SEED, num_agents=2):
        super(DDPGAgent, self).__init__()

        # number of agents
        self.num_agents = num_agents

        # single actor to play for both agents
        self.actor = Actor(state_size, action_size, random_seed).to(device)
        self.target_actor = Actor(state_size, action_size, random_seed).to(device)

        # single critic, critic input = states from all agents + actions from all agents
        self.critic = Critic(self.num_agents*state_size, self.num_agents*action_size, random_seed).to(device)
        self.target_critic = Critic(self.num_agents*state_size, self.num_agents*action_size, random_seed).to(device)

        # noise for exploration
        self.noise = OUNoise(action_size, scale=1.0)
        
        # initialize targets same as original networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.device = 'cpu'

        # replay buffer
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, SEED, self.device)

        self.samples_added = 0

    def reset(self):
        self.noise.reset()
        self.samples_added = 0

    def act(self, state, noise=0.0):
        """Returns actions for given state as per current policy."""
        actions = []
        # get an action for all the agents from the same actor, passing states observed by that agent
        for i in range(self.num_agents):
            input_state = torch.from_numpy(state[i]).float().to(device)
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(input_state).cpu().data.numpy()
            self.actor.train()
            if noise > 0:
                action += noise * self.noise.noise().cpu().data.numpy()
            actions.append(action)
        return np.clip(actions, -1, 1)

    def target_act(self, state, noise=0.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.target_actor.eval()
        with torch.no_grad():
            action = self.target_actor(state).cpu().data.numpy()
        self.target_actor.train()
        if noise > 0:
            action += noise * self.noise.noise()
        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        # add all experiences from agents in the same shared buffer replay
        self.buffer.add(states, actions, rewards, next_states, dones)
        self.samples_added += 1

        # Learn, if enough samples are available in the buffer
        if len(self.buffer) > BATCH_SIZE:
            self.learn()

    def learn(self):
        for i in range(self.num_agents):
            states, actions, rewards, next_states, dones = self.buffer.sample()

            # each agent gets next actions from the same actor with its own observed states
            actions_for_critic = []
            for k in range(self.num_agents):
                next_actions = self.target_actor(next_states[:, k])
                actions_for_critic.append(next_actions)

            with torch.no_grad():
                q_next = self.target_critic(torch.flatten(next_states, start_dim=1), torch.cat(actions_for_critic, dim=1)).squeeze(-1)

            y = rewards[:, i] + GAMMA * q_next * (1 - dones[:, i])

            q = self.critic(torch.flatten(states, start_dim=1), torch.flatten(actions, start_dim=1)).squeeze(-1)

            self.critic_optimizer.zero_grad()
            loss = torch.nn.HuberLoss()
            critic_loss = loss(q, y.detach())
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

            # each agent gets predicted actions from the same actor with its own observed states
            actions_for_critic = []
            for k in range(self.num_agents):
                actions_pred = self.actor(states[:, k])
                actions_for_critic.append(actions_pred)

            actor_loss = -self.critic(torch.flatten(states, start_dim=1), torch.cat(actions_for_critic, dim=1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        #if self.samples_added % 100 == 0:
        self.soft_update(self.target_critic, self.critic, TAU)
        self.soft_update(self.target_actor, self.actor, TAU)

    def hard_update(self, target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target_model, local_model, tau=TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, name='maddpg'):
        torch.save(self.target_actor.state_dict(), name+'_checkpoint_actor.pth')
        torch.save(self.target_critic.state_dict(), name+'_checkpoint_critic.pth')

    def load(self, name='maddpg'):
        self.critic.load_state_dict(torch.load(name+'_checkpoint_critic.pth'))
        self.actor.load_state_dict(torch.load(name+'_checkpoint_actor.pth'))
        self.target_critic.load_state_dict(torch.load(name+'_checkpoint_critic.pth'))
        self.target_actor.load_state_dict(torch.load(name+'_checkpoint_actor.pth'))

