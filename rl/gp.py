import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from env import StockEnv

class PG():
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 0.99
        self.episode_buffer = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model = self.build_continuous_policy_model()
        # self.model.load_weights("Weights\\MountainCarContinuous_PG\\pg_checkpoint")
        self.log_std = tf.Variable(tf.zeros(n_actions, dtype=tf.float32))

    def build_continuous_policy_model(self):
        lstm_input = tf.keras.layers.Input(shape=(30, 81))
        dense_input = tf.keras.layers.Input(shape=(4))
        
        x = tf.keras.layers.Convolution1D(64, 3, activation='relu')(lstm_input)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(32, dropout=0.2)(x)
        all_out_put = tf.keras.layers.Dense(self.n_actions)(tf.keras.layers.concatenate([x, dense_input]))
        
        model = tf.keras.models.Model(inputs=[lstm_input, dense_input], outputs=[all_out_put])
        return model

    def choose_action(self, observation1, observation2):
        mu = self.model([np.expand_dims(observation1, axis=0), np.expand_dims(observation2, axis=0)])
        normal_distribution = tfp.distributions.Normal(mu, tf.math.exp(self.log_std))
        sampled_action = normal_distribution.sample()
        return np.squeeze(sampled_action, axis=0)

    def store(self, state1, state2, action, reward):
        self.episode_buffer.append((state1, state2, action, reward))

    def get_store(self):
        states1, states2, actions, rewards = [], [], [], 
        for state1, state2, action, reward in self.episode_buffer:
            states1.append(state1)
            states2.append(state2)
            actions.append(action)
            rewards.append(reward)
        self.episode_buffer = []
        return np.array(states1), np.array(states2), np.array(actions), np.array(rewards)

    def discount_and_norm_rewards(self, rewards):
        # 计算折减奖励
        discounted = np.zeros((len(rewards)))
        discount_rewards = np.power(self.gamma, np.arange(0, len(rewards), dtype=np.float32))
        for t in range(len(rewards)):
            discounted[t] = np.sum(rewards[t:] * discount_rewards[0: len(rewards)-t])

        normalize_rewards = (discounted - np.mean(discounted)) / np.std(discounted)
        return normalize_rewards

    def train(self):
        states1, states2, actions, rewards = self.get_store()
        discounted_rewards = self.discount_and_norm_rewards(rewards)
        with tf.GradientTape() as tape:
            mu = self.model([states1, states2])
            normal_distribution = tfp.distributions.Normal(mu, tf.math.exp(self.log_std))
            log_probs = normal_distribution.log_prob(actions)
            loss = - tf.reduce_mean(log_probs * discounted_rewards)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


start_date = '2015-09-30'
end_date = '2021-10-01'
single_stock = '000001.XSHE'
env = StockEnv(start_date, end_date, single_stock=single_stock)
agent = PG(n_actions=env.action_space.shape[0], n_features=None)

i_episode = 0

while True:
    time_step = 0
    i_episode += 1

    observation1, observation2 = env.reset()

    while True:
        time_step += 1
        # env.render()

        action = agent.choose_action(observation1, observation2)

        observation, reward, done, _ = env.step(action)

        print(env.date, reward)

        agent.store(observation1, observation2, action, reward)

        if done:
            agent.train()
            # print("i_episode: {0} \t time_step: {1} \t max_episode_steps: {2}".format(i_episode, time_step, ))
            break

        observation1, observation2 = observation

    if i_episode != 0 and i_episode % 10 == 0:
        agent.model.save_weights("weight\\gp\\pg_checkpoint")
