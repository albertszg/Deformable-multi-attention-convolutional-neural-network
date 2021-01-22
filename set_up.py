import os
import time
import warnings
import torch
from torch import nn
from torch import optim
from CWRU import CWRU
from CWRUslice import CWRUSlice
from XJTUSLICE import XJTUSlice
from MFPTSlice import MFPTSlice
from CNN1d import CNN
from biLSTM import BiLSTM
from deformconv_lstm import DeformLSTM
from EA import EAconvLSTM
from SE import SEconvLSTM
from only_CNN import only_conv
from conv_lstm import convLSTM
from attention_deform_net import AD_convLSTM
# from tensorboardX import SummaryWriter
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import itertools
import random
import scipy.io

random.seed(666)
# 绘制matlabplot中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']


#  绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j].numpy(),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 计算混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1

    return conf_matrix


# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() is True else 'cpu')
print(device)
# 从数据集中提取样本输出
batch_size = 64
path = r'/media/ubuntu/Data/hcy/dl_based_bearing_diagnosis/tmp'
data = CWRUSlice(path, 'mean-std')

# path = r'/media/ubuntu/Data/hcy/XJTU-SY_Bearing_Datasets'
# data = XJTUSlice(path, 'mean-std')

# path = r'/media/ubuntu/Data/hcy/Mttp'
# data = MFPTSlice(path, 'mean-std')


datasets = {}
datasets['train'], datasets['val'] = data.data_preprare()


dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=8)
                for x in ['train', 'val']}

# 设置训练参数
cchannel=10
name1='AD-CWRU'
# net = SEconvLSTM(out_channel=cchannel).to(device)#channel attention
net = EAconvLSTM(out_channel=cchannel).to(device)#space attention
# net = convLSTM(out_channel=cchannel).to(device)#base
# net = DeformLSTM(out_channel=cchannel).to(device)#defrom conv
# net = AD_convLSTM(out_channel=cchannel).to(device)
# net = only_conv(out_channel=cchannel).to(device)
lr = 0.001
# optimizer = optim.Adam(net.parameters(), lr=lr,  weight_decay=1e-5)#
optimizer = optim.SGD(net.parameters(), lr=lr,  momentum=0.9,weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss().to(device)
# writer = SummaryWriter('runs/new_data2')

# training
step = 0
best_acc = 0.0
batch_count = 0
batch_loss = 0.0
batch_acc = 0
step_start = time.time()
max_epoch = 200
losses = np.ones(max_epoch)
acces = np.ones(max_epoch)
conf_matrix = torch.zeros(cchannel, cchannel)
# conf_matrix = torch.zeros(10, 10)
acc_val=[]
loss_val=[]

for epoch in range(max_epoch):
    print('-'*5 + 'Epoch {}/{}'.format(epoch, max_epoch - 1) + '-'*5)

    for phase in ['train', 'val']:
        epoch_start = time.time()
        epoch_acc = 0
        epoch_loss = 0.0

        if phase == 'train':
            net.train()
        else:
            net.eval()

        for idx, (input, label) in enumerate(dataloaders[phase]):
            inputs = input.to(device)
            labels = label.to(device)

            with torch.set_grad_enabled(phase == 'train'):
                # output,w1 = net(inputs)
                output = net(inputs)
                loss = loss_fn(output, labels)

                pred = output.argmax(dim=1)
                correct = torch.eq(pred, labels).float().sum().item()
                loss_temp = loss.item() * inputs.size(0)
                epoch_loss += loss_temp
                epoch_acc += correct

                if phase == 'train':
                    # writer.add_scalars('acc', {'train acc': epoch_acc / len(dataloaders[phase].dataset)}, epoch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss += loss_temp
                    batch_acc += correct
                    batch_count += inputs.size(0)

                    if step % 100 == 0:
                        batch_loss = batch_loss / batch_count
                        batch_acc = batch_acc / batch_count
                        temp_time = time.time()
                        train_time = temp_time - step_start
                        step_start = temp_time
                        batch_time = train_time / 100 if step != 0 else train_time
                        sample_per_sec = 1.0*batch_count/train_time
                        print('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                    '{:.1f} examples/sec {:.2f} sec/batch'.format(
                            epoch, idx*len(inputs), len(dataloaders[phase].dataset),
                            batch_loss, batch_acc, sample_per_sec, batch_time
                        ))
                        batch_acc = 0
                        batch_loss = 0.0
                        batch_count = 0
                    step += 1

                if phase == 'val':
                    # writer.add_scalars('acc', {'val acc': epoch_acc / len(dataloaders[phase].dataset)}, epoch)
                    step += 1
                    if epoch == max_epoch-1:
                        conf_matrix = confusion_matrix(output, labels, conf_matrix)

        epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
        epoch_acc = epoch_acc / len(dataloaders[phase].dataset)

        # writer.add_scalar('loss', epoch_loss, epoch)

        print('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
            epoch, phase, epoch_loss, phase, epoch_acc, time.time()-epoch_start
        ))
        if phase == 'train':
            print(type(epoch_loss))
            losses[epoch] = epoch_loss
            acces[epoch] = epoch_acc
        else:
             acc_val.append(epoch_acc)
             loss_val.append(epoch_loss)



        # save the model
        if phase == 'val':
            # save the checkpoint for other learning
            model_state_dic = net.state_dict()
            # save the best model according to the val accuracy
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                # w=w1
                # p1=dataloaders[phase].__iter__()
                # p=p1.__next__()

                print("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                # torch.save(model_state_dic,
                #             os.path.join('saved_model/exp3', '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

if cchannel==15:
   xjtu_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
else:
   xjtu_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# classes = ['normal', 'inner0.007', 'ball0.007', 'outer0.007', 'inner0.014', 'ball0.014', 'outer0.014', 'inner0.021', 'ball0.021', 'outer0.021']
# 绘制混淆矩阵、训练损失曲线、验证准确率曲线
# plot_confusion_matrix(conf_matrix, xjtu_classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
# plt.savefig('xjtu_result/conf_matrix_'+name1+'.jpg')
# plt.figure()
# plt.plot(np.arange(max_epoch),losses,color='black',label='train_loss')
# plt.plot(np.arange(max_epoch),loss_val,color='red',label='val_loss')
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('training loss')
# plt.savefig('xjtu_result/loss_'+name1+'.jpg')
# plt.figure()
# plt.plot(np.arange(max_epoch),acces,color='black',label='train_acc')
# plt.plot(np.arange(max_epoch),acc_val,color='red',label='val_acc')
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.savefig('xjtu_result/acc_'+name1+'.jpg')
# plt.show()
scipy.io.savemat('xjtu_result/'+name1+'_data.mat',{'train_acc':acces,'train_loss':losses,'val_acc':acc_val,'val_loss':loss_val})

print('max acc:{}'.format(max(acces)))
print('max val_acc:{}'.format(best_acc))
# print(w)
# print(w.shape)
# x=w[0]
# print(x[0].shape)
# print('______________________________________________________')
# print('______________________________________________________')
# # print(x[0])
# # plt.imshow(x[0].cpu(),'gray_r')
# # plt.xticks([])
# # plt.yticks([])
# # plt.savefig('xjtu_result/EA_attention.jpg')
# # plt.show()
# print('______________________________________________________')
# print('______________________________________________________')
# p2=p[0]
# print(p2.shape)
# for i in  range(64):
#     plt.imshow(p2[i][0], 'gray_r')
#     plt.xticks([])
#     plt.yticks([])
#     filename='xjtu_result/EA_attention_original_picture'+str(i)+'.jpg'
#     plt.savefig(filename)
# i=range(24)
# k=32
# plt.subplot(411)
# plt.plot(p2[k][0][3])
# plt.subplot(412)
# plt.plot(p2[k][0][4])
# plt.subplot(413)
# plt.plot(p2[k][0][5])
# plt.subplot(414)
# plt.plot(p2[k][0][6])
# plt.savefig('xjtu_result/EA_attention_part.jpg')
# plt.show()



