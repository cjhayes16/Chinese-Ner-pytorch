# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import argparse
from utils import built_train_dataset, built_dev_dataset, get_time_dif
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


parser = argparse.ArgumentParser(description='Chinese Ner Pytorch')
parser.add_argument('--doing', type=str, required=True, help='choose a action: train,test,predict')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert,Albert,Xlnet,Gpt-2')
args = parser.parse_args()


if __name__ == '__main__':

    model_name = args.model
    x = import_module('Models.' + model_name)
    config = x.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading Datas...")
    train_dataset = built_train_dataset(config)
    dev_dataset = built_dev_dataset(config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    if args.doing=='train':
        model = x.Model(config).to(config.device)
        train(config, model, train_dataset, dev_dataset)
    if args.doing=='predict':
        model = x.Model(config).to(config.device)
        predict(config,model,)
