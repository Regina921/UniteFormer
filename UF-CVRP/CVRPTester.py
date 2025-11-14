
import torch
import os
from logging import getLogger
from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # utility
        self.time_estimator = TimeEstimator()

    def run(self, xe_choice=None):
        self.time_estimator.reset()
        test_num_episode = self.tester_params['test_episodes']   
        self.env.load_raw_data(self.tester_params['test_episodes'])   
 
        # self.env.load_pkl_distribution_problems()  # 10000

        if xe_choice is None:
            xe_choice = self.tester_params['xe_choice']  # test 
        else:
            xe_choice = xe_choice   

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        optimal_reward_AM = AverageMeter()   
 
        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)   

        episode = 0
        loop_cnt = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score, optimal_reward = self._test_one_batch(episode, batch_size, xe_choice)   
            current_gap = (score - optimal_reward) / optimal_reward     
            aug_current_gap = (aug_score - optimal_reward) / optimal_reward   

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            optimal_reward_AM.update(optimal_reward, batch_size)  
            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            if loop_cnt <= 5:
                self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], opt_l:{:.3f}, score:{:.3f}, aug_score:{:.3f}, gap:{:4f}%, aug_gap:{:4f}%, XE_choice:{:1d}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, optimal_reward, score, aug_score, current_gap, aug_current_gap, xe_choice))
                loop_cnt += 1

            all_done = (episode == test_num_episode)
            if all_done:
                self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], opt_l:{:.3f}, score:{:.3f}, aug_score:{:.3f}, gap:{:4f}%, aug_gap:{:4f}%, XE_choice:{:1d}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, optimal_reward, score, aug_score, current_gap, aug_current_gap, xe_choice))
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" Distribution data: {} ".format(self.env_params['data_path']))
                self.logger.info(" XE_choice: {:1d} ".format(xe_choice))

                self.logger.info(" BEST SCORE: {:.4f} ".format(optimal_reward_AM.avg))
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

                gap = (score_AM.avg - optimal_reward_AM.avg) / optimal_reward_AM.avg * 100
                self.logger.info(" noAUG-Gap: {:.4f}%".format(gap))
                gap_aug = (aug_score_AM.avg - optimal_reward_AM.avg) / optimal_reward_AM.avg * 100
                self.logger.info(" AUG-Gap: {:.4f}%".format(gap_aug))

        return optimal_reward_AM.avg, score_AM.avg, aug_score_AM.avg, gap, gap_aug


    def _test_one_batch(self, episode, batch_size, xe_choice):
        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(episode, batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
 
            x_edges = self.env.x_edges           
            x_edges_values = self.env.x_edges_values   
 
            self.model.pre_forward(reset_state, x_edges, x_edges_values, xe_choice)
            self.optimal_reward = self.env._get_best_distance(self.env.problems, self.env.solution)
 
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item(), self.optimal_reward.item()  #.mean()
