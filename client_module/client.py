import os
import time
import argparse
import asyncio
import concurrent.futures

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import ClientConfig
from client_comm_utils import *
from training_utils import train, test
import utils
import datasets, models

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='VGG')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--visible_cuda', type=str, default='-1')


args = parser.parse_args()

if args.visible_cuda == '-1':
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(((int(args.idx) % 3)+1) % 2)
    if int(args.idx) < 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif int(args.idx) >= 2 and int(args.idx) < 8:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    elif int(args.idx) >= 8 and int(args.idx) < 14:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    elif int(args.idx) >= 14:
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip=args.master_ip,
        master_port=args.master_port
    )
    utils.create_dir("logs")
    recorder = SummaryWriter("logs/log_"+str(args.idx))
    # receive config
    master_socket = connect_send_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    # create model
    local_model = models.create_model_instance(args.dataset_type, args.model_type)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)

    # create dataset
    print("train data len : {}\n".format(len(client_config.custom["train_data_idxes"])))
    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    print("train dataset:")
    utils.count_dataset(train_loader)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, selected_idxs=client_config.custom["test_data_idxes"], shuffle=False)
    print("test dataset:")
    utils.count_dataset(test_loader)

    initial_lr = args.lr
    epoch_lr = initial_lr 
    min_lr = args.min_lr
    epoch = 1
    lr_schedule = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    lr_sc_idx = 1

    sc_idx = 0
    while True:
        print("--**--")
        if epoch >= 300:
            initial_lr = args.lr / 2
        if epoch > lr_schedule[lr_sc_idx]:
            lr_sc_idx += 1
        epoch_lr = min_lr + (initial_lr - min_lr) * 0.5 * (1+np.cos((epoch-lr_schedule[lr_sc_idx-1])/(lr_schedule[lr_sc_idx] - lr_schedule[lr_sc_idx-1])*np.pi))
        # epoch_lr = initial_lr

        print("epoch-{} lr: {}".format(epoch, epoch_lr))

        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, momentum=0.9, weight_decay=args.weight_decay)
        start_time = time.time()
        train_loss = train(local_model, train_loader, optimizer, device=device, model_type=args.model_type)
        print("train time: ", time.time() - start_time)
        local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
        
        test_loss, acc = test(local_model, test_loader, device, model_type=args.model_type)
        send_data_socket((epoch, acc, train_loss, test_loss, local_para), master_socket)
        recorder.add_scalar('acc_worker-' + str(args.idx), acc, epoch)
        recorder.add_scalar('test_loss_worker-' + str(args.idx), test_loss, epoch)
        recorder.add_scalar('train_loss_worker-' + str(args.idx), train_loss, epoch)
        print("epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
        print("\n\n")

        global_para, flag =  get_data_socket(master_socket)
        epoch = flag + 1
        torch.nn.utils.vector_to_parameters(global_para, local_model.parameters())

    torch.save(local_model.state_dict(), './logs/model_'+str(args.idx)+'.pkl')
    # close socket

    master_socket.shutdown(2)
    master_socket.close()



if __name__ == '__main__':
    main()
