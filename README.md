# Python-Package-for-Deep-Q-Learning

DQN implementation

DQN implementation is based on the paper:
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.


Mountain Car Domain

Mountain car is standard platform for testing RL algorithms in which a underpowered car tries to reach a goal position uphill by moving to and fro the hill valley. The state space of the car is continuous and consist of its position and velocity. At every state, it can choose out of 3 possible actions -- move forward, backward or stay. Refer to this <a href="https://en.wikipedia.org/wiki/Mountain_car_problem">Wikipedia Article</a> for more information.

**Files**
1. DQN.py
2. DQNTest.py
3. Utils.py
4. Final.py
5. Final1.py


**Training** <br/>
Run the file DQN.py

**Testing**<br/>
Run DQNTest.py file. Select the corresponding '.model' file.

**Dependencies**<br/>
Python3<br/>
numpy<br/>
Keras<br/>
Matplotlib
