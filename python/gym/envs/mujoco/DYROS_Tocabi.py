import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from math import exp, sin, cos, pi
import time
from pyquaternion import Quaternion

GroundCollisionCheckBodyList = ["base_link",\
            "R_HipRoll_Link", "R_HipCenter_Link", "R_Thigh_Link", "R_Knee_Link",\
            "L_HipRoll_Link", "L_HipCenter_Link", "L_Thigh_Link", "L_Knee_Link",\
            "Waist1_Link", "Waist2_Link", "Upperbody_Link", \
            "R_Shoulder1_Link", "R_Shoulder2_Link", "R_Shoulder3_Link", "R_Armlink_Link", "R_Elbow_Link", "R_Forearm_Link", "R_Wrist1_Link", "R_Wrist2_Link",\
            "L_Shoulder1_Link", "L_Shoulder2_Link", "L_Shoulder3_Link", "L_Armlink_Link", "L_Elbow_Link", "L_Forearm_Link", "L_Wrist1_Link","L_Wrist2_Link"]

SelfCollisionCheckBodyList = GroundCollisionCheckBodyList + ["L_AnkleCenter_Link", "L_AnkleRoll_Link", "L_Foot_Link", "R_AnkleCenter_Link", "R_AnkleRoll_Link", "R_Foot_Link"]

ObstacleList = ["obstacle1", "obstacle2", "obstacle3", "obstacle4", "obstacle5", "obstacle6", "obstacle7", "obstacle8", "obstacle9"]


class DYROSTocabiEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frameskip=8):
        mujoco_env.MujocoEnv.__init__(self, 'dyros_tocabi.xml', frameskip)
        utils.EzPickle.__init__(self)
        for id in GroundCollisionCheckBodyList:
            self.ground_collision_check_id.append(self.model.body_name2id(id))
        for id in SelfCollisionCheckBodyList:
            self.self_collision_check_id.append(self.model.body_name2id(id))
        self.ground_id.append(0)
        # for id in ObstacleList:
        #     self.ground_id.append(self.model.body_name2id(id))
        print("Collision Check ID", self.ground_collision_check_id)
        print("Self Collision Check ID", self.self_collision_check_id)
        print("Ground ID", self.ground_id)
        print("R Foot ID",self.model.body_name2id("R_Foot_Link"))
        print("L Foot ID",self.model.body_name2id("L_Foot_Link"))

    def _get_obs(self):

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        orientation = Quaternion(array=qpos[3:7])
        noise_angle = np.random.normal(0, 0.034,3)
        orientation = orientation * Quaternion(axis=(1.0, 0.0, 0.0), radians=noise_angle[0]) * \
                        Quaternion(axis=(0.0, 1.0, 0.0), radians=noise_angle[1]) * Quaternion(axis=(0.0, 0.0, 1.0), radians=noise_angle[2])
        return np.concatenate([orientation.elements.flatten(),
                            qpos[7:].flatten(),
                            qvel[6:].flatten()])


    def step(self, a):
        self.time += self.dt
        done_by_early_stop = False
        self.action_log.append(a)
        # print("Action: ", a)
        # a[:] = 0.0

        # Simulation
        for _ in range(self.frame_skip): 
            self.do_simulation(a,1)
            # self.render()

        # Collision Check
        for i in range(self.sim.data.ncon):
            if (any(self.model.geom_bodyid[self.sim.data.contact[i].geom1] == ground_id for ground_id in self.ground_id) and \
                    any(self.model.geom_bodyid[self.sim.data.contact[i].geom2] == collisioncheckid for collisioncheckid in self.ground_collision_check_id)) or \
                (any(self.model.geom_bodyid[self.sim.data.contact[i].geom2] == ground_id for ground_id in self.ground_id) and \
                    any(self.model.geom_bodyid[self.sim.data.contact[i].geom1] == collisioncheckid for collisioncheckid in self.ground_collision_check_id)):
                done_by_early_stop = True # Ground-Body contact
            if (any(self.model.geom_bodyid[self.sim.data.contact[i].geom1] == self_col_id for self_col_id in self.self_collision_check_id) and \
                    any(self.model.geom_bodyid[self.sim.data.contact[i].geom2] == self_col_id for self_col_id in self.self_collision_check_id)):
                done_by_early_stop = True # Self Collision contact

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        basequat = self.sim.data.get_body_xquat("Neck_Link")
        basequat_desired = np.array([1,0,0,0])  #self.mocap_data[self.mocap_data_idx,4:8]
        baseQuatError = (1-np.dot(basequat_desired,basequat))

        mimic_body_orientation_reward =  0.1 * exp(-200*baseQuatError)
        mimic_qpos_reward =  0.7*exp(-2.0*(np.linalg.norm(self.init_q_desired[7:] - qpos[7:])**2))
        mimic_qvel_reward =  0.2*exp(-0.002*(np.linalg.norm(self.init_qvel[6:] - qvel[6:])**2))
        mimic_base_pose_reward = 0.1*exp(-9.2*np.linalg.norm(self.init_q_desired[0:3] - qpos[0:3]))

        reward = mimic_body_orientation_reward + mimic_qpos_reward + mimic_qvel_reward + mimic_base_pose_reward

        if not done_by_early_stop:
            self.epi_len += 1
            self.epi_reward += reward
            if self.epi_len == 2000:
                print("Epi len: ", self.epi_len)
                np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')

                return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len), specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_base_pose_reward=mimic_base_pose_reward))

            return self._get_obs(), reward, done_by_early_stop, dict(specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_base_pose_reward=mimic_base_pose_reward))
        else:
            mimic_body_orientation_reward = 0.0
            mimic_qpos_reward = 0.0
            mimic_qvel_reward = 0.0
            mimic_base_pose_reward = 0.0
            reward = 0.0

            print("Epi len: ", self.epi_len)
            np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')

            return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len), specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_base_pose_reward=mimic_base_pose_reward))


    def reset_model(self):
        self.time = 0.0
        self.epi_len = 0
        self.epi_reward = 0

        self.set_state(self.init_q_desired, self.init_qvel,)  

        self.pert_duration = 0
        self.cur_pert_duration = 0

        self.action_log = []
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
