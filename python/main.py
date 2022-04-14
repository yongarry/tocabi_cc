import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

import numpy as np
import datetime
import torch as th
import time

def main():
   new_train = False

   file_name = "ppo2_DYROSTocabi_" + str(datetime.datetime.now())

   if (new_train):
      # multiprocess environment
      n_cpu = 8
      env = SubprocVecEnv([lambda: gym.make('DYROSTocabi-v1') for i in range(n_cpu)])
      env = VecNormalize(env, norm_obs=True, clip_obs=2.0, norm_reward=False, training=True)

      # n_cpu = 1   
      # env = gym.make('DYROSTocabi-v1')
      # env = DummyVecEnv([lambda: env])
      # env = VecNormalize(env, norm_obs=True, clip_obs=2.0, norm_reward=False, training=False)

      model = PPO('MlpPolicy', env, verbose=1, n_steps=int(4096/n_cpu), wandb_use=True)
      model.learn(total_timesteps=20000000)

      model.save(file_name)
      env.save(file_name+"_env.pkl")

      model.policy.to("cpu")
      for name, param in model.policy.state_dict().items():
         name= name.replace(".","_")
         weight_file_name = "./result/" + name + ".txt"
         np.savetxt(weight_file_name, param.data)

      np.savetxt("./result/obs_mean.txt", env.obs_rms.mean)
      np.savetxt("./result/obs_variance.txt", env.obs_rms.var)

      del model # remove to demonstrate saving and loading
      del env

   else:
      file_name = "ppo2_DYROSTocabi_2022-04-14 09:53:45.560911"
   
      env = gym.make('DYROSTocabi-v1')
      env = DummyVecEnv([lambda: env])
      env = VecNormalize.load(file_name+"_env.pkl", env)
      env.training = False

      model = PPO.load(file_name, env=env, wandb_use=False)

      model.policy.to("cpu")
      for name, param in model.policy.state_dict().items():
         name= name.replace(".","_")
         weight_file_name = "./result/" + name + ".txt"
         np.savetxt(weight_file_name, param.data)
      
      np.savetxt("./result/obs_mean.txt", env.obs_rms.mean)
      np.savetxt("./result/obs_variance.txt", env.obs_rms.var)
      #Enjoy trained agent
      obs =  np.copy(env.reset())

      epi_reward = 0
      while True:
         action, _states = model.predict(obs, deterministic=True)
         obs, rewards, dones, info = env.step(action)
         env.render()
         epi_reward += rewards      
         
         if dones:
            print("Episode Reward: ", epi_reward)
            epi_reward = 0
      
if __name__ == '__main__':
    main()