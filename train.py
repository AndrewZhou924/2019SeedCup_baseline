import torch
from config import Config
from model import Network,My_MSE_loss
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import time
from dataLoader import TrainSet,ValSet
from model import Network
from evaluation import calculateAllMetrics
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from tqdm import tqdm

opt = Config()

# prepare dataset
print("==> loading data...")
trainset = TrainSet(opt.TRAIN_FILE, opt=opt)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.TRAIN_BATCH_SIZE, shuffle=True)

valset = ValSet(opt.VAL_FILE, opt=opt)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.VAL_BATCH_SIZE, shuffle=False)
print("==> load data successfully")

# setup network
net = Network(opt)
if opt.USE_CUDA:
    print("==> using CUDA")
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True

# set criterion (loss function)
criterion_1 = torch.nn.MSELoss()
criterion_2 = My_MSE_loss()

# you can choose metric in [accuracy, MSE, RankScore]
highest_metrics = 100

def train(epoch):
    net.train()
    print("train epoch:", epoch)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.get_lr(epoch))
    for batch_idx, (inputs, targets_sign_day, targets_sign_hour, targets_ship_day, targets_ship_hour, targets_got_day, targets_got_hour, targets_dlved_day, targets_dlved_hour) in enumerate(tqdm(trainloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()
            targets_sign_day = targets_sign_day.cuda()
            targets_sign_hour = targets_sign_hour.cuda()
            targets_ship_day = targets_ship_day.cuda()
            targets_ship_hour = targets_ship_hour.cuda()
            targets_got_day = targets_got_day.cuda()
            targets_got_hour = targets_got_hour.cuda()
            targets_dlved_day = targets_dlved_day.cuda()
            targets_dlved_hour = targets_dlved_hour.cuda()

        inputs = torch.autograd.Variable(inputs)
        targets_sign_day = torch.autograd.Variable(targets_sign_day.float())
        targets_sign_hour = torch.autograd.Variable(targets_sign_hour.float())
        targets_ship_day = torch.autograd.Variable(targets_ship_day.float())
        targets_ship_hour = torch.autograd.Variable(targets_ship_hour.float())
        targets_got_day = torch.autograd.Variable(targets_got_day.float())
        targets_got_hour = torch.autograd.Variable(targets_got_hour.float())
        targets_dlved_day = torch.autograd.Variable(targets_dlved_day.float())
        targets_dlved_hour = torch.autograd.Variable(targets_dlved_hour.float())

        optimizer.zero_grad()

        (output_FC_1_1, output_FC_1_2) = net(inputs.float())

                
        output_FC_1_1 = output_FC_1_1.reshape(-1)
        # output_FC_2_1 = output_FC_2_1.reshape(-1)
        # output_FC_3_1 = output_FC_3_1.reshape(-1)
        # output_FC_4_1 = output_FC_4_1.reshape(-1)
        
        output_FC_1_2 = output_FC_1_2.reshape(-1)
        # output_FC_2_2 = output_FC_2_2.reshape(-1)
        # output_FC_3_2 = output_FC_3_2.reshape(-1)
        # output_FC_4_2 = output_FC_4_2.reshape(-1)

        loss_1_1 = criterion_2(output_FC_1_1, targets_sign_day)
        # loss_2_1 = criterion_1(output_FC_2_1, targets_ship_day)
        # loss_3_1 = criterion_1(output_FC_3_1, targets_got_day)
        # loss_4_1 = criterion_1(output_FC_4_1, targets_dlved_day)
        
        loss_1_2 = criterion_1(output_FC_1_2, targets_sign_hour)
        # loss_2_2 = criterion_1(output_FC_2_2, targets_ship_hour)
        # loss_3_2 = criterion_1(output_FC_3_2, targets_got_hour)
        # loss_4_2 = criterion_1(output_FC_4_2, targets_dlved_hour)

        loss_day  = loss_1_1
        loss_hour = loss_1_2

        loss = loss_day + loss_hour
        loss.backward()

        optimizer.step()

        # TODO add to tensorboard
        if batch_idx == 1:
            print("==> epoch {}: loss_day is {}, loss_hour is {} ".format(epoch, loss_day, loss_hour))

def val(epoch):
    global highest_metrics
    net.eval()
    pred_signed_time = []
    real_signed_time = []
    for batch_idx, (inputs, payed_time, signed_time) in enumerate(tqdm(valloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()

        inputs = torch.autograd.Variable(inputs)
        (output_FC_1_1, output_FC_1_2) = net(inputs.float())
        
        # calculate pred_signed_time via output
        for i in range(len(inputs)):

            pred_time_day = output_FC_1_1[i]
            pred_time_hour = output_FC_1_2[i]

            temp_payed_time = payed_time[i]
            temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")
            temp_payed_time = temp_payed_time.replace(hour = int(pred_time_hour)%24)

            temp_pred_signed_time = temp_payed_time + relativedelta(days = int(pred_time_day))
            temp_pred_signed_time = temp_pred_signed_time.replace(hour = int(pred_time_hour)%24)
            temp_pred_signed_time = temp_pred_signed_time.replace(minute = 0)
            temp_pred_signed_time = temp_pred_signed_time.replace(second = 0)
            # temp_pred_signed_time.

            pred_signed_time.append(temp_pred_signed_time.strftime("%Y-%m-%d %H"))
            real_signed_time.append(signed_time[i])

    (rankScore_result, onTimePercent_result, accuracy_result) = calculateAllMetrics(real_signed_time, pred_signed_time)
    print("==> epoch {}: rankScore is {}, onTimePercent is {}, accuracy is {}".format(epoch, rankScore_result, onTimePercent_result, accuracy_result))

    # save model
    if rankScore_result < highest_metrics:
        print("==> saving model")
        print("==> onTimePercent {} | rankScore {} ".format(onTimePercent_result, rankScore_result))
        highest_metrics = rankScore_result
        torch.save(net, opt.MODEL_SAVE_PATH)


# start training
if __name__=='__main__':
    for i in range(opt.NUM_EPOCHS):
        train(i)

        if i % opt.val_step == 0:
            val(i)
