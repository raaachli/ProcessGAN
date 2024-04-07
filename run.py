import os
import numpy as np
import yaml
import argparse
from nets.process_gan_time import ProcessGAN_Time
from datetime import datetime

# set the random seed
seed = 88
np.random.seed(seed)


def get_config(data, model):
    file_path = 'configurations/' + data + '_' + model + '.yaml'
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config['device'])
    return config


def run_gan_time(data, save_time, res_path_time, res_path_act, res_path_duration, mode, config, model, model_mode, gen_num ):
    save_path = 'result/' + data + '/' + save_time + '_' + mode  + '_'+model+str(model_mode)+'/'
    save_path_res = 'result/' + data + '/' + save_time + '_' + mode + '_'+model+str(model_mode)+'/' + 'stats/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_res), exist_ok=True)
    if model_mode == 1:
        gan = ProcessGAN_Time(res_path_time, res_path_act, res_path_duration, save_path, config, gen_num, 'Vanilla', model, model_mode)
    elif model_mode == 2:
        gan = ProcessGAN_Time(res_path_time, res_path_act, res_path_duration, save_path, config, gen_num, 'MSE', model, model_mode)
    elif model_mode == 3:
        gan = ProcessGAN_Time(res_path_time, res_path_act, res_path_duration, save_path, config, gen_num, 'MSE', model, model_mode)
    else:
        gan = ProcessGAN_Time(res_path_time, res_path_act, res_path_duration, save_path, config, gen_num, 'MSE', model, model_mode)
    gan.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=1, help='inclusion of auxiliary loss')
    # 1: no aux, 2: act, 3: time, 4: act+time
    parser.add_argument('--model', type=str, default='trans_attn', help='ProcessGAN model or transformer_gan model')
    # trans_attn: ProcessGAN model, discriminator with time-based attention.
    # trans: Transformer GAN model
    parser.add_argument('--data', type=str, default='SEP', help='dataset name')

    args = parser.parse_args()
    mode = args.mode
    o_model = args.model
    data = args.data

    ar_path = 'data/data_time/'+data+'/data_seq/'+data+'.txt'
    res_path = 'data/data_time/'+data+'/data_seq/'+data+'.txt'
    res_path_time = 'data/data_time/'+data+'/data_seq/'+data+'_time_dif_norm.txt'
    res_path_duration = 'data/data_time/'+data+'/data_seq/'+data+'_time_duration_norm.txt'

    dateTimeObjlocal = datetime.now()
    currDateTime = (
                "Received Timestamp: = " + str(dateTimeObjlocal.year) + str(dateTimeObjlocal.month) +str(
            dateTimeObjlocal.day) + str(dateTimeObjlocal.hour) + str(dateTimeObjlocal.minute) + str(
            dateTimeObjlocal.second) + "\n")
    print("System Timestamp: ", dateTimeObjlocal)

    save_time = str(dateTimeObjlocal.month) + str(dateTimeObjlocal.day) + str(dateTimeObjlocal.hour) + str(dateTimeObjlocal.minute)

    config = get_config(data, o_model)
    run_gan_time(data, save_time, res_path_time,  res_path, res_path_duration, 'Time', config,
                 model=o_model, model_mode=mode, gen_num=config['seq_num'])
