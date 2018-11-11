import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import model_fMRI as net
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import DataSet as myDataLoader
from torch.autograd import Variable
from clean import LabelInfo
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from load import loadData



def val(args, val_loader, model, criterion):
    model.eval()

    epoch_loss = []

    output_list = []
    label_list = []

    total_batches = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)

        output_list.extend(output.detach().data.cpu().numpy())
        label_list.extend(target.cpu().numpy().flatten().tolist())

        epoch_loss.append(loss.item())

        time_taken = time.time() - start_time

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))

    output_list = [np.argsort(np.abs(x))[-1] for x in output_list]

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    accuracy = accuracy_score(label_list, output_list)
    report = classification_report(label_list, output_list)

    return average_epoch_loss_val, accuracy, report




def train(args, train_loader, model, criterion, optimizer):
    # switch to train mode

    model.train()

    epoch_loss = []

    output_list = []
    label_list = []

    total_batches = len(train_loader)

    for i, (sequ, target) in enumerate(train_loader):
        start_time = time.time()

        if args.onGPU == True:
            sequ = sequ.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(sequ)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)

        optimizer.zero_grad()
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_list.extend(output.detach().data.cpu().numpy())
        label_list.extend(target.cpu().numpy().flatten().tolist())

        epoch_loss.append(loss.item())

        time_taken = time.time() - start_time

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))

    output_list = [np.argsort(x)[-1] for x in output_list]

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    accuracy = accuracy_score(label_list, output_list)
    report = classification_report(label_list, output_list)


    return average_epoch_loss_train, accuracy, report

def train_val(args):

    torch.set_default_tensor_type('torch.DoubleTensor')

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if args.visualizeNet == True:
        x = Variable(torch.randn(1, 51, 61, 23))

        if args.onGPU == True:
            x = x.cuda()

        model = net.ResNetC1()

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p

        print('Parameters: ' + str(total_paramters))


    logFileLoc = args.savedir + os.sep + args.trainValFile

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
        logger.write("%s\t%s\t\t\t\t\t%s\t\t\t%s\t\t\t%s\n" % ('Epoch', 'tr_loss', 'val_loss', 'tr_acc', 'val_acc'))
        logger.flush()
    else:
        logger = open(logFileLoc, 'w')
        logger.write("%s\t%s\t\t\t\t\t%s\t\t\t%s\t\t\t%s\n" % ('Epoch', 'tr_loss', 'val_loss', 'tr_acc', 'val_acc'))
        logger.flush()

    image, label = loadData()

    train_image, test_image, train_label, test_label = train_test_split(image, label, test_size=0.1,
                                                                    random_state=42, shuffle=True)
    train_image, val_image, train_label, val_label = train_test_split(train_image, train_label, test_size=0.1,
                                                                    random_state=42, shuffle=True)

    train_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(train_image, train_label),
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)
    val_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(val_image, val_label),
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    test_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(test_image, test_label),
                                                 batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.num_workers, pin_memory=True)

    model = net.ResNetC1()

    if args.onGPU == True:
        model = model.cuda()

    criteria = torch.nn.CrossEntropyLoss()

    if args.onGPU == True:
        criteria = criteria.cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=2e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    if args.onGPU == True:
        cudnn.benchmark = True

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.1)

    start_epoch = 0

    min_val_loss = 100

    for epoch in range(start_epoch, args.max_epochs):
        loss_train, accuracy_train, report_train = train(args, train_data_load, model, criteria, optimizer)
        loss_val, accuracy_val, report_val = val(args, val_data_load, model, criteria)

        logger.write("%s\t%s\t\t\t\t\t%s\t\t\t%s\t\t\t%s\n" % (epoch, loss_train, loss_val, accuracy_train, accuracy_val))

        alleleLoc = args.savedir + os.sep + 'acc_' + str(epoch) + '.txt'
        log = open(alleleLoc, 'a')
        log.write("train classification report")
        log.write("\n")
        log.write(report_train)
        log.write("\n")
        log.write("validation classification report")
        log.write("\n")
        log.write(report_val)
        log.flush()
        log.close()

        if loss_val < min_val_loss:
            if args.save_model == True:
                model_file_name = args.savedir + os.sep + 'best_model' + '.pth'
                print('==> Saving the best model')
                torch.save(model.state_dict(), model_file_name)
            min_val_loss = loss_val


    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--step_loss', type=int, default=20)
    parser.add_argument('--lr', type=float, default= 1e-3)
    parser.add_argument('--savedir', default='./results')
    parser.add_argument('--trainValFile', default='trainValFile.txt')
    parser.add_argument('--visualizeNet', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--onGPU', default=False)


    train_val(parser.parse_args())











