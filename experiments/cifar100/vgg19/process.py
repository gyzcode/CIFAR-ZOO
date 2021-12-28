import os, sys
import numpy as np

train_loss_txt = []
test_loss_txt = []
test_acc_txt  =[]
train_acc_txt  =[]


input = open('log＿alex_0.6.txt', 'r')

train_loss_iter = 0
train_loss = 0.0
train_acc = 0.0

for line in input:
    if 'test' in line and 'acc' in line:
        if not "==" in line[-8:-2]:
          test_acc_txt.append(float(line[-8:-2]))

    if 'test' in line and 'loss' in line:
        test_loss_txt.append(float(line[-26:-21]))

    if 'train' in line and 'loss' in line and 'step' in line:
        
        train_loss_iter += 1
        if train_loss_iter % 4 == 0:
            train_loss += float(line[-42:-36])
            train_loss /= 4.0
            train_loss_txt.append(train_loss)
            train_loss = 0.0
        else:
            train_loss += float(line[-42:-36])

    if 'train' in line and 'acc' in line and 'step' in line:
        #print(line[-23:-17])
        if train_loss_iter % 4 == 0:
            train_acc += float(line[-23:-17])
            train_acc /= 4.0
            train_acc_txt.append(train_acc)
            train_acc = 0.0
        else:
            train_acc += float(line[-23:-17])
        
np.savetxt("train_loss.txt", np.array(train_loss_txt))
np.savetxt("test_loss.txt", np.array(test_loss_txt))
np.savetxt("test_acc.txt", np.array(test_acc_txt))
np.savetxt("train_acc.txt", np.array(train_acc_txt))