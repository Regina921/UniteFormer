##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
##########################################################################################
# Path Config
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
##########################################################################################
# import
import logging
from utils import create_logger, copy_all_src
from CVRPTester import CVRPTester as Tester

##########################################################################################
# 0 edge 
# 1 node 
# 2 edge + node 
# XE_choice = 0 1 2

# parameters
test_paras = {
    # problem_size: [filename, episode, batch]
    20: ['vrp20_test_lkh_1w.txt', 10000, 100],
    50: ['vrp50_test_lkh_1w.txt', 10000, 1000],
    100: ['vrp100_test_lkh_1w.txt', 10000, 100],
 
    # 100: ['vrp100_test_lkh_128.txt', 128, 64],
    200: ['vrp200_test_lkh_128.txt', 128, 128],
    300: ['vrp300_test_lkh_128.txt', 128, 128],
    500: ['vrp500_test_lkh_128.txt', 128, 128],
}
b = os.path.abspath(".").replace('\\', '/')
XE_choice = 0  # 1 2
CVRP = 50   
NN = 20

env_params = {
    'mode': 'test',
    'problem_size': CVRP,
    'pomo_size': CVRP,
    'num_neighbors': NN,  
    'data_path': b + f"/data/{test_paras[CVRP][0]}",
    'optimal_label': None,
    # 'data_path': "./data/vrp20_test_lkh_1w.txt",  

}

model_params = {
    'encoder_layer_num': 3,

    'embedding_dim': 256,   
    'sqrt_embedding_dim': 256 ** (1 / 2),   
    'head_num': 16,  
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16 ** (1 / 2),   
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
 
    'GCN_dim': 3,   
    'mlp_layers': 3,
    'aggregation': "mean",
 
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1 / 2) ** (1 / 2),
    'ms_layer2_init': (1 / 16) ** (1 / 2),
}

tester_params = {
    'xe_choice': XE_choice,  # 0 1,2

    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './train_models/cvrp50',  

        'epoch': 1010,  #
    },
    'test_episodes': 10 * 1000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    'aug_factor': 8,   
    'aug_batch_size': 50,
    'test_data_load': {
        'enable': False,  # True
        'filename': './data/vrp20_1w.txt'
        # 'filename': './data/vrp100_test_seed1234.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {

        'desc': f'{CUDA_DEVICE_NUM}test1010_UFormer_N{NN}_cvrp50A8_{XE_choice}',  # 7

        'filename': 'log_T1w.txt'
    }
}


##########################################################################################
 
def main_test50(epoch, path, xe_choice, cuda_device_num=None):
    # epoch, self.result_folder, use_RRC=False
    if DEBUG_MODE:
        _set_debug_mode()
    if cuda_device_num is not None:
        tester_params['cuda_device_num'] = cuda_device_num
    create_logger(**logger_params)
    _print_config()
    tester_params['model_load'] = {
        'path': path,
        'epoch': epoch,
    }
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    copy_all_src(tester.result_folder)
    optimal_reward, score_optimal, aug_score_optimal, noaug_gap, aug_gap = tester.run(xe_choice)  

    return optimal_reward, score_optimal, aug_score_optimal, noaug_gap, aug_gap


# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
