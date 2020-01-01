# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from sklearn.metrics import f1_score, classification_report

from pytorch_pretrained import BertTokenizer
from pytorch_pretrained.optimization import *
from utils import *


def train(config, model, train_dataset, dev_dataset):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_dataset):
            spend_time=get_time_dif(start_time)
            model.zero_grad()
            model.train()
            batch= tuple(_.to(config.device) for _ in batch)
            input_id,input_mask,label,output_mask = batch
            encode=model(batch)
            loss=model.loss(encode,input_mask,label)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                print('step: {} |  epoch: {}|  loss: {:.4f}|  time: {}|'
                      .format(total_batch, epoch, loss.item(),spend_time))
            total_batch+=1
        dev(model,epoch,dev_dataset)
        torch.save(model.state_dict(),config.save_path)




def dev(model,epoch,dev_dataset):
    model.eval()
    count=0
    length = 0
    y_predicts,y_labels=[],[]
    eval_loss,eval_acc,eval_f1=0,0,0
    with torch.no_grad():
        for i,batch in enumerate(dev_dataset):
            input_ids, input_mask, label_ids, output_mask = batch
            length+=input_ids.size(0)
            encode = model(batch)
            eval_los = model.loss(encode,output_mask,label_ids)
            eval_loss = eval_los + eval_loss
            count += 1

            predicts = model.predict(encode, output_mask)
            y_predicts.append(predicts)

            label_ids = label_ids.view(1, -1)
            label_ids = label_ids[label_ids != -1]
            y_labels.append(label_ids)

        eval_predicted = torch.cat(y_predicts, dim=0)
        eval_labeled = torch.cat(y_labels, dim=0)
        model.acc_f1(eval_predicted, eval_labeled)
        model.class_report(eval_predicted, eval_labeled)
        print('eval  epoch : {}|  loss : {}'.format(epoch,eval_loss/length))


# def predict(model,config,input_str=""):
#     model.eval()  # 取消batchnorm和dropout,用于评估阶段
#     model.to(config.device)
#     VOCAB = config.bert_path  # your path for model and vocab
#     tokenizer = BertTokenizer.from_pretrained(VOCAB)
#     while True:
#         with torch.no_grad():
#                 input_str = input("请输入文本: ")
#                 input_ids = tokenizer.encode(input_str,add_special_tokens=True)  # add_spicial_tokens=True，为自动为sentence加上[CLS]和[SEP]
#                 input_mask = [1] * len(input_ids)
#                 output_mask = [0] + [1] * (len(input_ids) - 2) + [0]  # 用于屏蔽特殊token
#
#                 input_ids_tensor = torch.LongTensor(input_ids).reshape(1, -1)
#                 input_mask_tensor = torch.LongTensor(input_mask).reshape(1, -1)
#                 output_mask_tensor = torch.LongTensor(output_mask).reshape(1, -1)
#                 input_ids_tensor = input_ids_tensor.to(DEVICE)
#                 input_mask_tensor = input_mask_tensor.to(DEVICE)
#                 output_mask_tensor = output_mask_tensor.to(DEVICE)
#
#                 bert_encode = self.model(input_ids_tensor, input_mask_tensor)
#                 predicts = self.model.predict(bert_encode, output_mask_tensor)
#
#                 print('paths:{}'.format(predicts))
#                 entities = []
#                 for tag in self.tags:
#                     tags = get_tags(predicts[0], tag, self.model.tag_map)
#                     entities += format_result(tags, input_str, tag)
#                 print(entities)
