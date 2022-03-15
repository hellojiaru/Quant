import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from env import StockEnv


class PG:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 0.99
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(lr=0.01)
        self.episode_buffer = []

    def build_model(self):
        lstm_input = tf.keras.layers.Input(shape=(30, 81))
        dense_input = tf.keras.layers.Input(shape=(4))
        
        x = tf.keras.layers.Convolution1D(64, 3, activation='relu')(lstm_input)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(32, dropout=0.2)(x)
        all_out_put = tf.keras.layers.Dense(self.n_actions, activation="softmax")(tf.keras.layers.concatenate([x, dense_input]))
        
        model = tf.keras.models.Model(inputs=[lstm_input, dense_input], outputs=[all_out_put])
        return model

    def choose_action(self, observation1, observatio2):
        actions_prob = self.model([np.expand_dims(observation1, axis=0), np.expand_dims(observatio2, axis=0)])
        categorical_distribution = tfp.distributions.Categorical(probs=actions_prob)
        sampled_action = categorical_distribution.sample()
        return np.squeeze(sampled_action, axis=0)

    def store(self, state1, state2, action, reward):
        self.episode_buffer.append((state1, state2, action, reward))

    def get_store(self):
        states1, states2, actions, rewards = [], [], [], []
        for state1, state2, action, reward in self.episode_buffer:
            states1.append(state1)
            states2.append(state2)
            actions.append(action)
            rewards.append(reward)
        self.episode_buffer = []
        return np.array(states1), np.array(states2), np.array(actions), np.array(rewards)

    def train(self):
        states1, states2, actions, rewards = self.get_store()
        discounted_rewards = self.discount_and_norm_rewards(rewards)
        with tf.GradientTape() as tape:
            actions_prob = self.model([states1, states2])
            categorical_distribution = tfp.distributions.Categorical(probs=actions_prob)
            log_prob = categorical_distribution.log_prob(actions)
            loss = - tf.reduce_mean(log_prob * discounted_rewards)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def discount_and_norm_rewards(self, rewards):
        # 计算折减奖励
        discount_rewards = []
        discounted_sum = 0
        for reward in rewards:
            discounted_sum = discounted_sum * self.gamma + reward
            discount_rewards.append(discounted_sum)
        discount_rewards = discount_rewards[::-1]
        # 标准化奖励
        normalize_rewards = (discount_rewards - np.mean(discount_rewards)) / np.std(discount_rewards)
        return normalize_rewards


start_date = '2015-09-30'
end_date = '2021-10-01'
single_stock = '000001.XSHE'
env = StockEnv(start_date, end_date, single_stock=single_stock, allin=True)
agent = PG(n_actions=2, n_features=None)
agent.model.load_weights("weight\\gp_d\\gp_d_checkpoint")
i_episode = 0

while True:
    time_step = 0
    i_episode += 1

    observation1, observation2  = env.reset()

    while True:
        time_step += 1
        # env.render()

        action = agent.choose_action(observation1, observation2)

        observation, reward, done, _ = env.step(action)

        print(env.date, reward)

        agent.store(observation1, observation2, action, reward)

        if done:
            agent.train()
            # print("i_episode: {0} \t time_step: {1}".format(i_episode, time_step))
            break

        observation1, observation2 = observation
    
        if i_episode != 0 and i_episode % 10 == 0:
            agent.model.save_weights("weight\\gp_d\\gp_d_checkpoint")
