# MSS used for screen cap
from mss import mss
# sending commands
import pydirectinput
# Opencv allows us to frame processing
import cv2
# Transformational framework
import numpy as np
# OPR for game over extraction
import pytesseract
# visualize captured frames
from matplotlib import pyplot as plt
# bringing in time for pauses
import time
# environment components
from gym import Env
from gym.spaces import Box, Discrete
# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker
#import the DQN algorithm
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

class WebGame(Env):
    #setup the environment action and observation shapes
    def __init__(self):
        #subclass model
        super().__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.done_location = {'top': 405, 'left': 630, 'width': 660, 'height': 70}

    #what is called to do somthing in the game
    def step(self, action):
        #action key - 0= Space , 1=Duck(down) , 2 = No Action(no op) 
        action_map = {
            0:'space',
            1: 'down', 
            2: 'no_op'
        }
        if action !=2:
            pydirectinput.press(action_map[action])
        #checking whether the game is done
        done, done_cap = self.get_done() 
        #get the next observation
        observation = self.get_observation()
        #reward - we get a point for every frame we're alive
        reward = 1 
        #info dictionary
        info = {}
        return observation, reward, done, info
    #visualizing the game 
    def render(self):
        cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
    #this close down the observation
    def close(self):
        cv2.destroyAllWindows()    
     #restart the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()
    #get the part of observation of the game that we want
    def get_observation(self):
        #get screen capture of game
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100,83))
        channel = np.reshape(resized, (1,83,100))
        return channel 
    #get the done text
    def get_done(self):
        #get done screen
        done_cap = np.array(self.cap.grab(self.done_location))
        #valid done text
        done_strings = ['GAME', 'GAHE']
        done=False
        # if np.sum(done_cap) < 44300000:
        #     done = True
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done, done_cap
'''

env.reset()

obs=env.get_observation()
image=cv2.cvtColor(obs[0], cv2.COLOR_GRAY2BGR)
plt.imshow(image)
plt.show()


don, done_cap=env.get_done()
plt.imshow(done_cap)
plt.show()


for episode in range(2): 
    obs = env.reset()
    done = False  
    total_reward   = 0
    while not   done: 
        obs, reward,  done, info =  env.step(env.action_space.sample())
        total_reward  += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward)) 
'''
env = WebGame()
#creat callback
env_checker.check_env(env)
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
callback = TrainAndLoggingCallback(check_freq=100, save_path=CHECKPOINT_DIR)
#creat the DQN model
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=10000, learning_starts=1000)
model.learn(total_timesteps=100000, callback=callback) 
 
#test out model
for episode in range(5): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
   