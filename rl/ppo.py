import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from env import StockEnv


class Actor(tf.keras.layers.Layer):
    def __init__(self, action_size, epsilon):
        super(Actor, self).__init__()
        self.epsilon = epsilon
        self.action_size = action_size
        self.dense = tf.keras.layers.Dense(action_size, activation=tf.nn.tanh)
        self.log_std = tf.Variable(tf.zeros(action_size, dtype=tf.float32))

    def call(self, inputs):
        mu = self.dense(inputs)
        log_std = self.log_std
        return mu, log_std

    def loss(self, inputs, advantages, actions, logp_old):
        mu, log_std = self(inputs)
        normal = tfp.distributions.Normal(mu, tf.exp(log_std))
        logp = tf.reduce_sum(normal.log_prob(actions), axis=-1, keepdims=True)
        ratio = tf.exp(logp - logp_old)

        pi_loss = tf.reduce_mean(tf.minimum(ratio * advantages, tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages))
        approx_kl = 0.5 * tf.reduce_mean(tf.square(logp - logp_old))
        entropy_loss = tf.reduce_mean(tf.reduce_sum(normal.entropy(), axis=-1))
        return pi_loss, entropy_loss, tf.reduce_mean(logp_old), tf.reduce_mean(logp), approx_kl, ratio


class Critic(tf.keras.layers.Layer):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        value = self.dense(inputs)
        return value

    def loss(self, inputs, returns):
        value = self(inputs)
        loss = 0.5 * tf.reduce_mean((returns - value) ** 2)
        return loss


class LSTM(tf.keras.layers.Layer):
    def __init__(self):
        super(LSTM, self).__init__()
        self.cnn = tf.keras.layers.Convolution1D(64, 3, activation='relu')
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)
        self.lstm2 = tf.keras.layers.LSTM(32, dropout=0.2)

    def call(self, inputs1, inputs2):
        x = self.cnn(inputs1)
        x = self.lstm(x)
        x = self.lstm2(x)
        x = tf.keras.layers.concatenate([x, inputs2])
        return x


class PPO(tf.keras.models.Model):
    def __init__(self, action_size, epsilon, entropy_reg, value_coeff, learning_rate, max_grad_norm):
        super(PPO, self).__init__()
        self.actor = Actor(action_size, epsilon)
        self.critic = Critic()
        self.value_coeff = value_coeff
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.initial_layer = LSTM()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs1, inputs2):
        x = self.initial_layer(inputs1, inputs2)
        mu, log_std = self.actor(x)
        value = self.critic(x)
        normal = tfp.distributions.Normal(mu, tf.exp(log_std))
        pi = tf.squeeze(normal.sample(1), axis=0)
        # TODO maybe apply action clipping
        logp_pi = tf.reduce_sum(normal.log_prob(pi), axis=-1, keepdims=True)
        return pi, logp_pi, value

    def loss(self, states1, states2, actions, returns, advantages, logp_old):
        with tf.GradientTape() as tape:
            x = self.initial_layer(states1, states2)
            value_loss = self.critic.loss(x, returns)
            pi_loss, entropy_loss, old_neg_log_val, neg_log_val, approx_kl, ratio = self.actor.loss(x, advantages, actions, logp_old)
            loss = - pi_loss - entropy_loss * self.entropy_reg + value_loss * self.value_coeff
        grads = tape.gradient(loss, self.trainable_weights)
        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return pi_loss, value_loss, entropy_loss, loss, old_neg_log_val, neg_log_val, approx_kl, ratio



########## hyperparameters ##########
GAMMA = 0.99
LAMBDA = 0.95

ACTION_SIZE = 1
EPSILON = 0.2
STEPS_PER_EPOCH = 512
STEPS_PER_BATCH = 128
EPOCHS = 10000000
TRAIN_K_MINIBATCH = 4

LEARNING_RATE = 3e-4
ENTROPY_REG = 0.01
VALUE_COEFFICIENT = 0.5

MAX_GRAD_NORM = 0.5


def compute_gae(rewards, values, bootstrap_values, dones, gamma, lam):
    values = np.vstack((values, [bootstrap_values]))
    deltas = []
    for i in reversed(range(len(rewards))):
        V = rewards[i] + (1.0 - dones[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))

    A = deltas[-1, :]
    advantages = [A]
    for i in reversed(range(len(deltas) - 1)):
        A = deltas[i] + (1.0 - dones[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    advantages = np.array(list(advantages))
    return advantages


def update(ppo, states1, states2, actions, returns, advantages, old_log_prob):
    
    # states1 = states1.reshape([-1,30,81])
    # states2 = states2.reshape([-1,2])
    # actions = actions.reshape([-1,1])
    # returns = returns.reshape([-1,1])
    # advantages = advantages.reshape([-1,1])
    # old_log_prob = old_log_prob.reshape([-1,1])
    
    
    # states1: 10, 237, 30, 81
    # states2: 10, 237, 2

    loss = 0
    indexs = np.arange(states1.shape[0])
    for i in range(TRAIN_K_MINIBATCH):
        # np.random.shuffle(indexs)
        for start in range(0, len(indexs), 237):
            end = start + 237
            mbinds = indexs[start:end]
            slices = (arr[mbinds] for arr in (states1, states2, actions, returns, advantages, old_log_prob))
            pi_loss, value_loss, entropy_loss, total_loss, old_neg_log_val, neg_log_val, approx_kl, ratio = ppo.loss(*slices)

            loss += total_loss.numpy()

    return loss


def train():
    start_date = '2015-09-30'
    end_date = '2021-10-01'
    single_stock = '000001.XSHE'
    env = StockEnv(start_date, end_date, single_stock=single_stock)
    ppo = PPO(ACTION_SIZE, EPSILON, ENTROPY_REG, VALUE_COEFFICIENT, LEARNING_RATE, MAX_GRAD_NORM)
    # ppo.load_weights("weight\\ppo\\ppo_checkpoint")
    
    steps = 0
    timesteps = 0
    total_reward = 0
    
    observation1, observation2 = env.reset()
    
    for epoch in range(EPOCHS):
        states1, states2, actions, values, rewards, dones, old_log_pi = [], [], [], [], [], [], []

        for t in range(int(STEPS_PER_EPOCH)):
            
            
            input1 = np.expand_dims(observation1, axis=0)
            input2 = np.expand_dims(observation2, axis=0)
            
            pi, old_log_p, v = ppo.call(input1, input2)
            
            # observation, reward, done, _ = env.step(np.clip(pi.numpy(), -1, 1))
            observation, reward, done, _ = env.step(pi.numpy())
            
            observation1, observation2 = observation

            total_reward += reward
            
            print(env.date, reward)

            states1.append(observation1)
            states2.append(observation2)
            actions.append(pi.numpy()[0])
            values.append(v.numpy()[0])
            rewards.append(reward)
            dones.append(done)
            old_log_pi.append(old_log_p.numpy()[0])

            if done :
                observation1, observation2 = env.reset()
                total_reward = 0
        
        input1 = np.expand_dims(observation1, axis=0)
        input2 = np.expand_dims(observation2, axis=0)
        pi, old_log_p, v = ppo.call(input1, input2)
        last_val = v.numpy()[0]

        advantages = compute_gae(rewards, values, last_val, dones, GAMMA, LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        loss = update(ppo, np.array(states1), np.array(states2), np.array(actions), np.array(returns), np.array(advantages), np.array(old_log_pi))

        print("total_loss：{}   total_reward：{}".format(loss, total_reward))
        if epoch != 0 and epoch % 10 == 0:
            ppo.save_weights("weight\\ppo\\ppo_checkpoint")

if __name__ == '__main__':
    train()
