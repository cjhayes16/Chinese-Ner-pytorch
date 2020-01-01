# coding: UTF-8
import torch
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import TensorDataset, DataLoader
import datetime



PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # padding符号, bert中句子开头，结尾符号


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask,output_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask
        self.output_mask = output_mask

def build_dataset(config):

    def load_dataset(data_path,label_path,pad_size,label_dic,tokenizer):
        """
        :param data_path:文本数据路径
        :param label_path:标签数据路径
        :param pad_size:每个句子最大长度
        :param label_dic:词性种类表
        :return:
        """

        result=[]
        with open(data_path, 'r', encoding='utf-8') as df:
            with open(label_path, 'r', encoding='utf-8') as lf:
                train_data=df.readlines()
                train_label=lf.readlines()
                for word , label in zip(train_data,train_label):
                    tokens=word.split()
                    label=label.split()

                    if len(tokens) > pad_size - 2:  # 大于最大长度进行截断
                        tokens = tokens[0:(pad_size - 2)]
                        label = label[0:(pad_size - 2)]

                    # token to index
                    tokens_c_s = '[CLS] ' + ''.join(tokens) + ' [SEP]'
                    label_c_s = ' '.join(label)

                    tokenized_text=tokenizer.tokenize(tokens_c_s)
                    input_ids=tokenizer.convert_tokens_to_ids(tokenized_text)

                    label_ids=[label_dic[i] for i in label_c_s.split()]
                    input_mask=[1]*len(input_ids)

                    if len(input_ids) < pad_size:
                        input_id = input_ids + ([0]*(pad_size-len(input_ids)))
                        input_mask += ([0]*(pad_size-len(input_ids)))
                    if len(label_ids) < pad_size:
                        label_id = label_ids+([-1]*(pad_size-len(label_ids)))

                    output_mask=[1]*len(tokens)
                    output_mask=[0]+output_mask+[0]
                    if len(output_mask) < pad_size:
                        output_mask +=([0]*(pad_size-len(output_mask)))

                    assert len(input_id) == pad_size
                    assert len(input_mask) == pad_size
                    assert len(label_id) == pad_size
                    assert len(output_mask) == pad_size

                    #              处理后数据
                    # -------------------------------------------
                    # 原始:           我 是 中 国 人
                    # 分词:     [CLS] 我 是 中 国 人 [SEP]
                    # input_id:  101  2 12 13 16 14  102  0  0  0
                    # input_mask:  1  1  1  1  1  1    1  0  0  0
                    # label_id:       T  T  0  0  0   -1 -1 -1 -1 -1
                    # output_mask: 0  1  1  1  1  1    0  0  0  0

                    feature = InputFeatures(input_id=input_id,label_id=label_id,input_mask=input_mask,output_mask=output_mask)
                    result.append(feature)
        return  result
    data=load_dataset(config.train_data_path,config.train_label_path,config.pad_size,config.vocab_class,config.tokenizer)
    train_ids = torch.LongTensor([_.input_id for _ in data])
    train_masks = torch.LongTensor([_.input_mask for _ in data])
    train_tags = torch.LongTensor([_.label_id for _ in data])
    output_masks = torch.LongTensor([_.output_mask for _ in data])
    train_dataset = TensorDataset(train_ids, train_masks, train_tags, output_masks)
    return train_dataset

def built_train_dataset(config):
    train_dataset=build_dataset(config)
    return DataLoader(train_dataset,shuffle=True,batch_size=config.batch_size)


def built_dev_dataset(config):
    dev_dataset=build_dataset(config)
    return DataLoader(dev_dataset,shuffle=False,batch_size=config.batch_size)

# def save_model(model, epoch, path='result', **kwargs):
#     """
#     默认保留所有模型
#     :param model: 模型
#     :param path: 保存路径
#     :param loss: 校验损失
#     :param last_loss: 最佳epoch损失
#     :param kwargs: every_epoch or best_epoch
#     :return:
#     """
#     if not os.path.exists(path):
#         os.mkdir(path)
#     if kwargs.get('name', None) is None:
#         cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
#         name = cur_time + '--epoch:{}'.format(epoch)
#         full_name = os.path.join(path, name)
#         torch.save(model.state_dict(), full_name)
#         print('Saved model at epoch {} successfully'.format(epoch))
#         with open('{}/checkpoint'.format(path), 'w') as file:
#             file.write(name)
#             print('Write to checkpoint')
#
#
# def load_model(model, path='result', **kwargs):
#     if kwargs.get('name', None) is None:
#         with open('{}/checkpoint'.format(path)) as file:
#             content = file.read().strip()
#             name = os.path.join(path, content)
#     else:
#         name=kwargs['name']
#         name = os.path.join(path,name)
#     model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
#     print('load model {} successfully'.format(name))
#     return model


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


