import argparse
import os
import numpy as np
import yaml
from nets.process_gan_time import ProcessGAN_Time
import utils.helper as helper

# Set the random seed.
seed = 88
np.random.seed(seed)


def get_config(data: str, model: str) -> dict:
    file_path = os.path.join('configurations', data + '_' + model + '.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config['device'])
    return config


def run_gan_time(config: dict) -> None:
    save_time = config['save_time']
    save_path = os.path.join('result', config['data'], f'{save_time}/')
    save_path_res = os.path.join(save_path, 'stats/')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_res), exist_ok=True)
    config['save_path'] = save_path
    gan = ProcessGAN_Time(config)
    gan.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='act_time_loss', help='Choose from different auxiliary losses.')
    parser.add_argument('--model', type=str, default='trans_attn', help='Select model to use as generator. ProcessGAN or GAN with two transformers.')
    parser.add_argument('--data', type=str, default='SEP', help='Choose the dataset.')
    args = parser.parse_args()

    ar_path = os.path.join('data', 'data_time', args.data, 'data_seq', f'{args.data}.txt')
    res_path = os.path.join('data', 'data_time', args.data, 'data_seq', f'{args.data}.txt')
    res_path_time = os.path.join('data', 'data_time', args.data, 'data_seq', f'{args.data}_time_dif_norm.txt')
    res_path_duration = os.path.join('data', 'data_time', args.data, 'data_seq', f'{args.data}_time_duration_norm.txt')

    save_time = helper.get_timestamp()
    config = get_config(args.data, args.model)
    config['data'] = args.data
    config['model_name'] = args.model
    config['model_mode'] = args.mode
    config['save_time'] = save_time
    config['res_path_time'] = res_path_time
    config['res_path'] = res_path
    config['res_path_duration'] = res_path_duration
    config['res_path_duration'] = res_path_duration

    run_gan_time(config)
