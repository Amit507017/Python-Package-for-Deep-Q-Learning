import json
import numpy as np
import keras
from keras.models import model_from_json
import gym
import utils

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    state_space = env.observation_space
    action_space = env.action_space
    model=keras.models.load_model('Final1.model')
    model.compile("sgd","mse")
    utils.displayValueFunction(model=model, res=25, env=gym.make("MountainCar-v0"))
    for e in range(10):
        c=0
        done=False
        curr_state=env.reset().reshape(1, 2)

        while not done:
            Q_future=model.predict(curr_state)
            action=np.argmax(Q_future[0])
            next_state, reward, done,_ = env.step(action)
            env.render()
            c+=1

        if c>=199:
            success='Test Failed'
        else:
            success='Test Passed'

        print("Episodes %d, Steps %d "%(e,c)+ success)