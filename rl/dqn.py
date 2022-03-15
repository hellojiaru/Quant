import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from env import StockEnv


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = 0.003
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.buffer = deque(maxlen=1000)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model = self.build_model()

    def build_model(self):
        lstm_input = tf.keras.layers.Input(shape=(30, 81))
        dense_input = tf.keras.layers.Input(shape=(4))
        
        x = tf.keras.layers.Convolution1D(64, 3, activation='relu')(lstm_input)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(32, dropout=0.2)(x)
        all_out_put = tf.keras.layers.Dense(self.action_size)(tf.keras.layers.concatenate([x, dense_input]))
        
        model = tf.keras.models.Model(inputs=[lstm_input, dense_input], outputs=[all_out_put])
        return model

    def choose_action(self, state1, state2):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if np.random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.model.predict([np.expand_dims(state1, axis=0), np.expand_dims(state2, axis=0)]))
        return action

    def memorize(self, state1, state2, action, reward, done, next_state1, next_state2):
        self.buffer.append((state1, state2, action, reward, done, next_state1, next_state2))
        return len(self.buffer) >= self.batch_size

    def train(self):
        sample_batch = random.sample(self.buffer, self.batch_size)
        for state1, state2, action, reward, done, next_state1, next_state2 in sample_batch:
            with tf.GradientTape() as tape:
                target = reward if done else reward + self.gamma * self.model([np.expand_dims(next_state1, axis=0), np.expand_dims(next_state2, axis=0)]).numpy().max()
                y_true = self.model([np.expand_dims(state1, axis=0), np.expand_dims(state2, axis=0)]).numpy()
                y_true[0][action] = target
                loss = tf.keras.losses.mse(y_true,  self.model([np.expand_dims(state1, axis=0), np.expand_dims(state2, axis=0)]))
            gradient = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))


i_episode = 0
start_date = '2015-09-30'
end_date = '2021-10-01'
single_stock = '000001.XSHE'
env = StockEnv(start_date, end_date, single_stock=single_stock, allin=True)
state_size = None
action_size = 2
agent = DQN(state_size, action_size)

while True:

    i_episode += 1
    step_time = 0

    state1, state2 = env.reset()

    while True:
        step_time += 1
        # env.render()

        action = agent.choose_action(state1, state2)

        observation, reward, done, _ = env.step(action)
        
        next_state1, next_state2 = observation
        
        if agent.memorize(state1, state2, action, reward, done, next_state1, next_state2):
            agent.train()

        state1 = next_state1
        state2 = next_state2

        if done:
            break
        print(env.date, reward)

    if i_episode != 0 and i_episode % 50 == 0:
        agent.model.save_weights("weight\\dqn\\dqn_checkpoint")
