import getopt
import sys
import random
import numpy as np
import pickle
import copy
from collections import deque
from hanabi_learning_environment import rl_env
from tensorflow.keras import models, layers

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
        self.learning_rate = 0.001

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

    def select_action(self, observation, Q_network, input_data_size, epsilon=0.0):
        curr_player_index = observation["current_player"]
        # Select action based on epsilon-greedy strategy
        if np.random.rand() < epsilon:
            # Exploration: choose a random action
            legal_actions = observation["player_observations"][curr_player_index]["legal_moves"]
            return np.random.choice(len(legal_actions)), True
        else:
            # Exploitation: choose the best action (max Q-value)
            input_data = np.expand_dims(np.array(observation["player_observations"][curr_player_index]["vectorized"]),axis=0)
            # input_data = input_data.reshape(-1, 1, 755)  # Adding a sequence dimension
            input_data = input_data.reshape(-1, 1, input_data_size)  # Adding a sequence dimension


            Q_values = Q_network.predict(input_data)  # Predict Q-values for all actions
            #action = int(np.argmax(Q_values))  # Choose the action with the highest Q-value
            # print(action)
            actions_list = (-Q_values).argsort()
            #print(actions_list)
            actions_list = actions_list[0][0]
            for action in actions_list:
                if action in observation["player_observations"][curr_player_index]["legal_moves_as_int"]:
                    action_index = observation["player_observations"][curr_player_index]["legal_moves_as_int"].index(action)
                    #print(np.expand_dims(np.array(observation["player_observations"][curr_player_index]["vectorized"]),axis=0).shape)
                    action_index = observation["player_observations"][curr_player_index]["legal_moves_as_int"].index(action)
                    return action_index, False
            print("something's not right")


    # def select_action(self, observation, epsilon=0.1): # returns index of legal moves as int (action index)
    #     # Epsilon-greedy policy: explore or exploit
    #     self.epsilon = epsilon
    #     cur_player_index = observation["current_player"]
    #     if random.random() < self.epsilon:
    #         # print(observation['player_observations'][observation['current_player']]['legal_moves_as_int'])
    #         return (random.randint(0, len(observation['player_observations'][cur_player_index]['legal_moves_as_int']) - 1), True)  # Exploration
    #     else:
    #         # Exploit: choose the action with the highest Q-value for the current observation
    #         action_q_values = [self.get_q(observation, a) for a in observation['player_observations'][cur_player_index]['legal_moves_as_int']]
    #         # print(action_q_values)
    #         return (int(np.argmax(action_q_values)), False)

def initialize_network(input_shape, output_size):
    """
    Initializes the Q-network (a neural network model).
    
    Args:
    - input_shape: Tuple representing the shape of the input (state space).
    - output_size: Number of possible actions (output layer size).
    
    Returns:
    - The initialized Q-network model.
    """
    model = models.Sequential()
    
    # Add layers to the network
    model.add(layers.Dense(128, input_shape=input_shape, activation='relu'))  # Hidden layer 1
    model.add(layers.Dense(128, activation='relu'))  # Hidden layer 2
    model.add(layers.Dense(output_size, activation='linear'))  # Output layer, one Q-value per action
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')  # Mean Squared Error for Q-value loss
    
    return model


def train_agent(environment, agent, input_data_size, n_episodes=1000, num_players=2, print_legal_moves=False):
    max_reward = 0
    gamma = 0.1
    batch_size = 64
    buffer_size = 100  # Example size, adjust as needed
    target_update_frequency = 1000
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 0.95
    replay_buffer = deque(maxlen=buffer_size)  # Fixed-size replay buffer
    # Q_networks = [initialize_network((1, 755), environment.num_moves()) for _ in range(num_players)]
    Q_networks = [initialize_network((1, input_data_size), environment.num_moves()) for _ in range(num_players)]

    target_networks = [copy.deepcopy(Q_networks[0]) for _ in range(num_players)]
    total_rewards = []

    for episode in range(n_episodes):
        # Reset the environment and initialize variables for this episode
        observation = environment.reset()  # Initial observation
        total_reward = 0
        done = False
        score = 0
        while not done:
            for player_index in range(num_players):
                agent = agents[player_index]
                Q_network = Q_networks[player_index]
                target_network = target_networks[player_index]
                cur_player_index = observation["current_player"]
                if observation["player_observations"][player_index]["current_player_offset"] == 0: # play only when it is a player's turn
                    action_index, eps_true = agent.select_action(observation, Q_network,input_data_size, epsilon=epsilon)
                    action = observation["player_observations"][cur_player_index]["legal_moves"][action_index]
                    # print(f"Player {player_index+1}, epsiloned: {eps_true},action: {action}")

                    if print_legal_moves:
                        print(f"Legal actions:")
                        for act in observation['player_observations'][cur_player_index]['legal_moves']:
                            print(act)

                    # Apply action to the environment and get next observation, reward, and done signal
                    next_observation, reward, done, _ = environment.step(action)
                    score = max(score, environment.state.score())
                    # print(f"reward: {reward}")    
                    replay_buffer.append((observation['player_observations'][cur_player_index]['vectorized'], 
                                          observation['player_observations'][cur_player_index]['legal_moves_as_int'][action_index], 
                                          reward, 
                                          done, 
                                          next_observation['player_observations'][cur_player_index]['vectorized']))
                    # print(len(replay_buffer))
                    if len(replay_buffer) >= batch_size:
                        batch = random.sample(replay_buffer, batch_size)
                        observations, actions, rewards, dones, next_observations = zip(*batch)
                        
                        # Compute target Q-values for the mini-batch
                        states = np.array(next_observations)
                        next_Q_values = target_network.predict(np.expand_dims(np.array(states),axis=1))
                        target_Q_values = np.array(rewards).reshape(-1,1) + gamma * np.max(next_Q_values, axis=2) * (1 - np.array(dones).reshape(-1,1))
                        target_Q_values_one_hot = np.zeros((64, 20))

                        # For each observation, set the Q-value for the selected action
                        for i in range(64):
                            target_Q_values_one_hot[i, actions[i]] = target_Q_values[i]  # Replace with the target Q-value for the selected action
                        Q_network.train_on_batch(np.expand_dims(np.array(observations), axis=0), target_Q_values_one_hot)
                    # Update Q-table based on the agent's experience
                    if episode % target_update_frequency == 0:
                        target_networks[player_index].set_weights(Q_network.get_weights())
                    
                        # Decay epsilon over time
                        epsilon = max(epsilon_min, epsilon * epsilon_decay)

                    # Move to the next state
                    observation = next_observation
                    total_reward += reward
        total_rewards.append(total_reward)
        # scores.append(score)    
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")
    print(max_reward)
    return Q_networks, [total_rewards]

def evaluate_agent(environment, agents, input_data_size, target_networks, n_episodes=100, num_players=2, epsilon=0.0, print_legal_moves=False):
    total_rewards = [[], []]
    scores = []

    # Set epsilon to 0 for evaluation (no exploration)

    for episode in range(n_episodes):
        # Reset the environment and initialize variables for this episode
        observation = environment.reset()  # Initial observation
        total_reward = 0
        score = 0
        done = False
        
        while not done:
            for player_index in range(num_players):
                cur_player_index = observation["current_player"]
                agent = agents[player_index]
                if observation["player_observations"][player_index]["current_player_offset"] == 0:  # Player's turn
                    Q_network = target_networks[player_index]

                    # Choose the best action (greedy action) based on Q-values
                    action_index, eps_true = agent.select_action(observation, Q_network, input_data_size)
                    action = observation["player_observations"][cur_player_index]["legal_moves"][action_index]
                    print(f"action:{action}")
                    # print(observation["player_observations"][cur_player_index]['card_knowledge'])
                    # print("observed cards:")
                    # for card in observation['player_observations'][cur_player_index]['observed_hands']:
                    #     print(card)
                    if print_legal_moves:
                        print(f"Legal actions:")
                        for act in observation['player_observations'][cur_player_index]['legal_moves']:
                            print(act)

                    # Apply action to the environment and get next observation, reward, and done signal
                    next_observation, reward, done, _ = environment.step(action)
                    # print(f"reward:{reward}")
                    score = max(environment.state.score(), score)

                    # Update total reward for this episode
                    total_reward += reward

                    # Move to the next state
                    observation = next_observation

        total_rewards.append(total_reward)
        scores.append(score)
        print(f"Evaluation Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}, Total Score: {score}")

    # Calculate the average reward across all episodes
    avg_score = np.mean(scores)
    print(f"Average Score over {n_episodes} evaluation episodes: {avg_score}")

    return scores

import matplotlib.pyplot as plt

def generate_graphs(training_scores, agent_labels=["Lorem 1", "Ipsum 2", "Dolor 3"], scores=[[20, 22, 19, 23, 25, 21, 24],[18, 20, 19, 21, 22, 20, 19],[15, 17, 16, 18, 14, 16, 19]]):
    ### Agent Evaluation Histogram
    for score_list in scores:
        # bins = list(range(27))  # 0-26 for edges to include 25 bins
        bins = np.arange(start=0, stop=2, step=0.2)
        plt.hist(score_list, bins=bins, edgecolor='black', align='left')

        plt.xlabel('Points Scored')
        plt.ylabel('Frequency')

        plt.title('Agent Performance over 100 Evaluation Games')

        plt.show()

    ### Performance Box Plots
    plt.figure(figsize=(8, 6))
    plt.boxplot(scores, labels=agent_labels, patch_artist=True)

    plt.title("Agent Performance Comparison", fontsize=14)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    ### Model Training Graph
    episodes = np.arange(1, 1001)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, training_scores[0], label='Score', color='b')
    plt.plot(episodes, training_scores[1], label='Score', color='b')


    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.ylim(0, 12)  # Limit the y-axis from 0 to 25
    plt.xlim(1, 200)  # Limit the x-axis from 1 to 1000

    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    num_players = 2
    n_episodes = 200

    # -------------------------- TRAINING FOR FULL ----------------------------------------
    
    picklename = 'Qs_full_final.pkl'
    environment = rl_env.make('Hanabi-Full', num_players=num_players)

    # initial_state = environment.reset()
    # agents = [QLearningAgent(epsilon=0.1) for _ in range(num_players)]
    # Q_networks, training_scores = train_agent(environment, agents, input_data_size=658, n_episodes=n_episodes, num_players=num_players)
    # with open(picklename,'wb') as file:
    #     pickle.dump(Q_networks, file)
    # with open("training_scores_final.pkl", 'wb') as file:
    #     pickle.dump(training_scores, file)

    # -------------------------- EVALUATION FOR FULL -----------------------------------------

    training_scores1 = None
    with open("training_scores_final.pkl", 'rb') as file:
        training_scores1 = pickle.load(file)
    Q_networks = None
    with open(picklename, 'rb') as file:
        Q_networks = pickle.load(file)
    newagents = [QLearningAgent(epsilon=0) for _ in range(num_players)]
    scores1 = evaluate_agent(environment, newagents, 658, Q_networks, num_players=num_players, n_episodes=20)
    

    # --------------------------- TRAINING FOR INF -------------------------------------------

    picklename = 'Qs_inflife_final.pkl'
    environment = rl_env.make('Hanabi-Inf', num_players=num_players)

    # initial_state = environment.reset()
    # agents = [QLearningAgent(epsilon=0.1) for _ in range(num_players)]
    # Q_networks, training_scores = train_agent(environment, agents, input_data_size=755, n_episodes=n_episodes, num_players=num_players)
    # with open(picklename,'wb') as file:
    #     pickle.dump(Q_networks, file)
    # with open("training_scores_final_inf.pkl", 'wb') as file:
    #     pickle.dump(training_scores, file)

    # --------------------------- EVALUATION FOR INF ------------------------------------------
    training_scores2 = None
    with open("training_scores_final_inf.pkl", 'rb') as file:
        training_scores2 = pickle.load(file)
    Q_networks = None
    with open(picklename, 'rb') as file:
        Q_networks = pickle.load(file)
    newagents = [QLearningAgent(epsilon=0) for _ in range(num_players)]
    scores2 = evaluate_agent(environment, newagents, 755, Q_networks, num_players=num_players, n_episodes=20)
    
    # ----------------------------- GENERATE GRAPH --------------------------------------------
    
    scores = [scores1, scores2]
    print(scores2)
    generate_graphs(training_scores=[training_scores1, training_scores2], agent_labels=['Hanabi-Full, 100 Epochs', 'Hanabi-Inf, 100 Epochs'], scores=scores)

    # 1. Bar graph of EVALUATION SCORES (not rewards) between

    # 2. Box plot of EVALUATION SCORES (not rewards)

    # 3. Line graph of TRAINING REWARDS
    # Have 2 DIFFERENT ones
    # - One for FULL
    # - One for INF