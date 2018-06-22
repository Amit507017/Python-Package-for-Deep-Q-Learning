import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import gym
import random
import utils


class DQNAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.005
        self.batch_size = 32
        self.gamma = 0.95
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.target_model=self.build_model()

    def build_model(self):
        """Neural Network for Q-Learning"""
        model = Sequential()
        state_shape = self.state_space.shape[0]
        model.add(Dense(128, input_dim=state_shape, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.summary()
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Storing Transitions"""
        if len(self.memory)>1999:
            del self.memory[0]
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon Greedy Strategy"""
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            target = self.target_model.predict(state)

            if not done:
                next_state = next_state.reshape(1, 2)
                Q_pred = np.amax(self.target_model.predict(next_state))
                target[0][action] = reward + self.gamma * Q_pred
            else:
                target[0][action] = reward
            # training model with target
            self.model.fit(state, target, epochs=1, verbose=0)
            #utils.displayValueFunction(model=self.model,res=25,env=gym.make("MountainCar-v0"))

    def target_train(self):
        """Updating weights of target model"""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main():
    env = gym.make("MountainCar-v0")
    state_space = env.observation_space
    action_space = env.action_space
    trials = 5000
    trial_seq = 200

    agent = DQNAgent(state_space, action_space)

    success_count=0
    for trial in range(1, trials):
        env.seed()

        cur_state = env.reset().reshape(1, 2)
        print(cur_state)
        for stp in range(1, trial_seq):
            #env.render()
            action = agent.act(cur_state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, 2)
            agent.remember(cur_state, action, reward, next_state, done)
            agent.replay()
            if stp % 10 == 0:
                agent.target_train()
            cur_state = next_state
            if done:
                break

        print(stp)
        if stp >= 199:
            print("failed to converge in {} trial".format(trial))
            if trial % 100 == 0:
                agent.save_model('Final1.model')

        else:
            print("converged in {} trial".format(trial))
            success_count +=1
            agent.save_model('Final.model')

    print(success_count)

if __name__ == "__main__":
    main()