import getopt
import sys
import random
import numpy as np
import hanabi_learning_environment
import pickle
from hanabi_learning_environment import rl_env

# IMPORTANT DOCUMENTATION NOTE

# ACTION_ID = observation['player_observations'][observation['current_player']]['legal_moves_as_int'] (aka the LEGAL MOVE REPRESENTED AS AN INT BY HLE)
# ACTION_INDEX is the INDEX of the ACTION ID in the LIST OF LEGAL MOVES 
# ACTION = observation['player_observations'][observation['current_player']]['legal_moves'] (aka the LEGAL MOVE ITSELF, and is a DICT)

# EXAMPLE: LEGAL_MOVES_AS_INT = [12,13,14,15]
# 12 is an ACTION ID, and CORRESPONDING ACTION INDEX is 0
# corresponding ACTION would be osmething like {'action_type': 'PLAY', 'card_index': 0}

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        # self.n_actions = n_actions  # Number of possible actions (in Hanabi, this depends on the action space)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.Q = {}  # Q-table (dict to store Q-values for observation-action pairs)
        self.observation_map = {}

    def observation_to_id(self, observation):
        observation_tuple = tuple(observation['player_observations'][observation['current_player']]['vectorized'])
        if observation_tuple not in self.observation_map:
            self.observation_map[observation_tuple] = len(self.observation_map)
        return self.observation_map[observation_tuple]

    def get_q(self, observation, action):
        # Return Q-value for a given observation and action
        action_id = action
        observation_id = self.observation_to_id(observation)
        return self.Q.get((observation_id, action_id), 0.0)

    def update_q(self, observation, action_index, reward, next_observation):
        # Update Q-value using Q-learning formula
        cur_player_index = observation['current_player']
        action_id = observation['player_observations'][cur_player_index]['legal_moves_as_int'][action_index]
         # Exploit: choose the action with the highest Q-value for the current observation  
        action_q_values = [self.get_q(observation, a) for a in observation['player_observations'][cur_player_index]['legal_moves_as_int']]
        next_action_id = observation['player_observations'][observation['current_player']]['legal_moves_as_int'][int(np.argmax(action_q_values))]
        observation_id = self.observation_to_id(observation)
        current_q = self.get_q(observation, action_id)
        next_q = self.get_q(next_observation, next_action_id)

        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

        self.Q[(observation_id, action_id)] = new_q

    def select_action(self, observation, epsilon=0.1): # returns index of legal moves as int (action index)
        # Epsilon-greedy policy: explore or exploit
        self.epsilon = 0.1
        cur_player_index = observation["current_player"]
        if random.random() < self.epsilon:
            #print(observation['player_observations'][observation['current_player']]['legal_moves_as_int'])
            return (random.randint(0, len(observation['player_observations'][cur_player_index]['legal_moves_as_int']) - 1), True)  # Exploration
        else:
            # Exploit: choose the action with the highest Q-value for the current observation
            action_q_values = [self.get_q(observation, a) for a in observation['player_observations'][cur_player_index]['legal_moves_as_int']]
            print(action_q_values)
            return (int(np.argmax(action_q_values)), False)


def train_agent(environment, agent, n_episodes=1000, num_players=2, print_legal_moves=False):
    max_reward = 0
    
    for episode in range(n_episodes):
        # Reset the environment and initialize variables for this episode
        observation = environment.reset()  # Initial observation
        total_reward = 0
        done = False

        while not done:
            for player_index in range(num_players):
                agent = agents[player_index]
                cur_player_index = observation["current_player"]
                if observation["player_observations"][player_index]["current_player_offset"] == 0: # play only when it is a player's turn
                    action_index, eps_true = agent.select_action(observation)
                    action = observation["player_observations"][cur_player_index]["legal_moves"][action_index]
                    print(f"Player {player_index+1}, epsiloned: {eps_true},action: {action}")

                    if print_legal_moves:
                        print(f"Legal actions:")
                        for act in observation['player_observations'][cur_player_index]['legal_moves']:
                            print(act)

                    # Apply action to the environment and get next observation, reward, and done signal
                    next_observation, reward, done, _ = environment.step(action)

                    # Update Q-table based on the agent's experience
                    agent.update_q(observation, action_index, reward, next_observation)

                    # Move to the next state
                    observation = next_observation
                    total_reward += reward
        if max_reward < total_reward:
            max_reward = total_reward         
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")
    print(max_reward)

def evaluate_agent(environment, agent, n_episodes=100):
    total_rewards = 0
    for episode in range(n_episodes):
        observation = environment.reset()
        done = False
        total_reward = 0
        while not done:
            for player_index in range(num_players):
                agent = agents[player_index]
                cur_player_index = observation["current_player"]
                if observation["player_observations"][player_index]["current_player_offset"] == 0: # play only when it is a player's turn
                    action_index, eps_true = agent.select_action(observation)
                    action = observation["player_observations"][cur_player_index]["legal_moves"][action_index]
                    next_observation, reward, done, _ = environment.step(action)
                    observation = next_observation
                    total_reward += reward
                    
        
        total_rewards += total_reward
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")
    
    avg_reward = total_rewards / n_episodes
    print(f"Average Reward: {avg_reward}")

if __name__ == "__main__":

    num_players = 2
    n_episodes = 2

    environment = rl_env.make('Hanabi-Full', num_players=num_players)
    initial_state = environment.reset()
    agents = [QLearningAgent(epsilon=0.1) for _ in range(num_players)]

    #print(f"Q table at beginning: {agents[0].Q}")

    # train_agent(environment, agents, n_episodes=n_episodes, num_players=num_players)
    # picklename = 'agents_eps_'+str(n_episodes)+'_players_'+str(num_players)+'.pkl'
    # for agent in agents:
    #     print(len(agent.Q.keys()))
    # # with open(picklename,'wb') as file:
    # #     pickle.dump(agents, file)

    newagents = None
    with open("agents_eps_100000_players_2.pkl", 'rb') as file:
        newagents = pickle.load(file)
        for agent in newagents:
            print(agent.Q[max(agent.Q)])
    #print(f"Q table at end: {agents[0].Q}")
    
    evaluate_agent(environment, newagents, n_episodes=50)