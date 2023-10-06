import torch 
import random 
import numpy as np 
from collections import deque
from snake_game import SnakeGameAI, Direction, Point

# importing this after making the skeleton for the model.py
from model import LinearQNet, QTrainer

# importing after creating the helper.py (for plotting the results)
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # Learning Rate 

class Agent : 

    def __init__(self) :
        self.n_games = 0
        self.epsilon = 0        # parameter to control the randomness 
        self.gamma = 0.8          # discount rate (for the deep Q learning)
        self.memory = deque(maxlen=MAX_MEMORY)      #if exceeds, it'll call popleft()

        # TO DO : model, trainer 
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game) : 
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # List with all the 11 states of the snake head : 
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):    # done = game over state
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached 
        # note that we are appending (storing) as a single tuple

    def train_long_memory(self) :
        if len(self.memory) > BATCH_SIZE : 
            mini_sample  = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else : 
            mini_sample = self.memory
        
        # combining all the diffirent tuples values together using the zip function for the trainer()
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) :
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state) : 
        # Doing some random moves : trade-off between exploration and exploitation 
        self.epsilon = 80-self.n_games
        final_move = [0,0,0]

        # This way the value of epsilon will keep decreasing with the increase in number of games , thus there will be lesser random moves 
        if random.randint(0,200) < self.epsilon : 
            move = random.randint(0,2)
            final_move[move] = 1
        else : 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 


def train() :
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 
    agent = Agent()
    game = SnakeGameAI()

    # out Training loop 
    while True:
        # get the old (current state)
        state_old = agent.get_state(game)

        #get move (based on the current (old) state)
        final_move = agent.get_action(state_old)

        # perform the move and get the new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train the short memory of the agent 
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember the above and store it to remember it 
        agent.remember(state_old, final_move, reward, state_new, done)

        if done : 
            # Train the long memory (replay memory or exprience memory). Retrains on the moves taken during the previously completed game 
            # also want to plot the results 
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()

            # checking for a new highscore 
            if score > record :  
                record = score
                agent.model.save()

            print('Game',agent.n_games, 'Score : ', score, 'highscore : ', record)

            plot_scores.append(score)
            total_score += score 
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__' : 
    train()
