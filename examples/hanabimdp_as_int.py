import getopt
import sys
import random
import numpy as np
import hanabi_learning_environment
from hanabi_learning_environment import rl_env

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        # self.n_actions = n_actions  # Number of possible actions (in Hanabi, this depends on the action space)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.Q = {}  # Q-table (dict to store Q-values for observation-action pairs)
        self.action_map = {}  # Map action dictionaries to unique IDs
        self.observation_map = {}

    def to_tuple(lst):
        return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

    def observation_to_id(self, observation):
        #print(sorted(observation.items()))
        observation_tuple = tuple(observation['player_observations'][observation['current_player']]['vectorized'])
        if observation_tuple not in self.observation_map:
            self.observation_map[observation_tuple] = len(self.observation_map)
        return self.observation_map[observation_tuple]

    def get_q(self, observation, action):
        # Return Q-value for a given observation and action
        action_id = action
        observation_id = self.observation_to_id(observation)
        return self.Q.get((observation_id, action_id), 0.0)

    def update_q(self, observation, action, reward, next_observation):
        # Update Q-value using Q-learning formula
        action_id = action
         # Exploit: choose the action with the highest Q-value for the current observation  
        action_q_values = [self.get_q(observation, a) for a in observation['player_observations'][observation['current_player']]['legal_moves_as_int']]
        next_action_id = observation['player_observations'][observation['current_player']]['legal_moves_as_int'][int(np.argmax(action_q_values))]
        observation_id = self.observation_to_id(observation)

        current_q = self.get_q(observation, action)
        next_q = self.get_q(next_observation, next_action_id)

        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

        self.Q[(observation_id, action_id)] = new_q

    def select_action(self, observation):
        # Epsilon-greedy policy: explore or exploit
        if random.random() < self.epsilon:
            #print(observation['player_observations'][observation['current_player']]['legal_moves_as_int'])
            return random.randint(0, len(observation['player_observations'][observation['current_player']]['legal_moves_as_int']) - 1)  # Exploration
        else:
            # Exploit: choose the action with the highest Q-value for the current observation
            action_q_values = [self.get_q(observation, a) for a in observation['player_observations'][observation['current_player']]['legal_moves_as_int']]
            #print(type(np.argmax(action_q_values)))
            return int(np.argmax(action_q_values)) 


def train_agent(environment, agent, n_episodes=1000):
    max_reward = 0
    for episode in range(n_episodes):
        # Reset the environment and initialize variables for this episode
        observation = environment.reset()  # Initial observation
        total_reward = 0
        done = False
        #max_reward = 0
        while not done:
            #print(observation['player_observations'])
            action_index = agent.select_action(observation)
            cur_player_index = observation["current_player"]
            action = observation["player_observations"][cur_player_index]["legal_moves"][action_index]
            #print(action)

            # Apply action to the environment and get next observation, reward, and done signal
            next_observation, reward, done, _ = environment.step(action)
            #print("updating Q!")  
            #print("---------------reward-------------------")          
            #print()
            # Update Q-table based on the agent's experience
            # next_action should be the action with the highest action_q_values from the legal_moves
            # implement it in update_q

            agent.update_q(observation, action_index, reward, next_observation)
            #print("Q update finished")
            print(agent.Q)
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
            action = agent.select_action(observation)
            next_observation, reward, done, _ = environment.step(action)
            observation = next_observation
            total_reward += reward
        
        total_rewards += total_reward
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")
    
    avg_reward = total_rewards / n_episodes
    print(f"Average Reward: {avg_reward}")

if __name__ == "__main__":
	flags = {'players': 2, 'num_episodes': 1, 'agent_class': 'SimpleAgent'}
	options, arguments = getopt.getopt(sys.argv[1:], '',['players=','num_episodes=','agent_class='])
	environment = rl_env.make('Hanabi-Full')
	# initial_state = environment.reset()
	agent = QLearningAgent()
	train_agent(environment, agent, n_episodes=100)
	evaluate_agent(environment, agent, n_episodes=100)










            

