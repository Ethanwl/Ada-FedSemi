import os
import sys
import time
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import math
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from config import *
from communication_module.comm_utils import *
from control_algorithm.greedy import EpsilonGreedy, EpsilonGreedyCost
from training_module import datasets, models, utils

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model_type', type=str, default='VGG')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--tf_batch_size', type=int, default=128)
parser.add_argument('--data_pattern', type=float, default=2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--transfer_lr', type=float, default=0.05)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--thd_avg', type=float, default=0.5)
parser.add_argument('--client_frac', type=float, default=0.5)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--worker_num', type=int, default=10)
parser.add_argument('--start_label', type=int, default=100)
parser.add_argument('--moving_global', type=int, default=1)
parser.add_argument('--start_select', type=int, default=1)

parser.add_argument('--importance', type=int, default=1)
parser.add_argument('--hyper_idx', type=int, default=0)
parser.add_argument('--prob', type=float, default=1.0)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SERVER_IP = "127.0.0.1"

def main():
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    # init config
    common_config = CommonConfig()
    common_config.master_listen_port_base += random.randint(0, 30) * 31
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.ratio = args.prob
    common_config.epoch = args.epoch
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.weight_decay = args.weight_decay

    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)
    worker_num = args.worker_num

    # create workers
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list'][:worker_num]):
        custom = dict()
        killport(common_config.master_listen_port_base+worker_idx)
        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       client_ip=worker_config['ip_address'],
                                       master_ip=SERVER_IP,
                                       master_port=common_config.master_listen_port_base+worker_idx,
                                       custom = custom),
                   common_config=common_config, 
                   user_name=worker_config['user_name'],
                   pass_wd=worker_config['pass_wd'],
                   local_scripts_path=workers_config['scripts_path']['local'],
                   remote_scripts_path=workers_config['scripts_path']['remote'],
                   location='local'
                   )
        )
    # hyper parameters
    hyperpara_list = [
                    [20, 100, 20],
                    [50, 50, 10],
                    [50, 100, 10],
                    [10, 100, 20],
                    [70, 50, 10],
                    [70, 100, 10]
                ]
    hyperpara = hyperpara_list[args.hyper_idx]
    print("hyper parameters", hyperpara)
    # init decision maker
    search_interval = 10
    client_frac_table = np.linspace(0.1, 1, search_interval)
    sample_thd_table = np.linspace(0.80, 0.98, search_interval)
    agent_client = EpsilonGreedyCost(search_interval, client_frac_table, hyperpara[0])
    agent_sample = EpsilonGreedy(search_interval, sample_thd_table, hyperpara[1])

    # Create model instance
    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    teacher_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)

    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model Size: {} MB".format(model_size))

    # partition dataset
    predefined_data_pattern_cifar10 = [ [0.7, 0.15, 0.1],
                                        [0.7, 0.12, 0.1],
                                        [0.7, 0.10, 0.1],
                                        [0.7, 0.07, 0.1],
                                        [0.7, 0.05, 0.1],
                                        [0.7, 0.02, 0.1],

                                       [0.75, 0.15, 0.05],
                                       [0.75, 0.12, 0.05],
                                       [0.75, 0.10, 0.05],
                                       [0.75, 0.07, 0.05],
                                       [0.75, 0.05, 0.05],
                                       [0.75, 0.02, 0.05],

                                       [0.78, 0.15, 0.02],
                                       [0.78, 0.12, 0.02],
                                       [0.78, 0.10, 0.02],
                                       [0.78, 0.07, 0.02],
                                       [0.78, 0.05, 0.02],
                                       [0.78, 0.02, 0.02],
                                    ]
    if common_config.dataset_type == "CIFAR10":
        server_ratio, main_ratio, validation_ratio = predefined_data_pattern_cifar10[int(args.data_pattern)]
    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, server_ratio, main_ratio, validation_ratio, worker_num=worker_num)

    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=256, shuffle=False)

    for worker_idx, worker in enumerate(common_config.worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    print("Server unlabeled dataset")
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, shuffle=False, selected_idxs=train_data_partition.use(worker_num))
    count_dataset_total(train_loader)
    print("Server validation dataset")
    validation_loader = datasets.create_dataloaders(train_dataset, batch_size=256, shuffle=False, selected_idxs=train_data_partition.use(worker_num+1))
    count_dataset_total(validation_loader)

    # connect socket and send init config
    communication_parallel(common_config.worker_list, action="init")
    
    training_recorder = TrainingRecorder(common_config.worker_list, common_config.recoder)

    initial_lr = args.transfer_lr
    min_lr = 0.001
    selected_clients = list(range(worker_num))
    train_loss_mv = np.zeros((worker_num,))
    selected_c_num = len(selected_clients)
    beta = 0.9
    client_local_paras = list()
    p_client = np.ones((worker_num,)) / worker_num
    agg_weight = np.ones((worker_num,)) * (1.0 / worker_num)

    # lr_schedule = [0, 10, 30, 70, 150, 310, 630, 1270, 2550]
    lr_schedule = [0, 100, 200, 300, 400, 500, 600, 700, 800]
    lr_sc_idx = 1

    client_frac = 1.0
    client_frac_idx = 9
    last_epoch_val_loss = 0.0
    last_epoch_val_acc = 0.1

    training_recorder.recorder.add_scalar('Selected-num', 0, 0)
    training_recorder.recorder.add_scalar('Right-num', 0, 0)
    training_recorder.recorder.add_scalar('TfLoss', 0, 0)

    for epoch_num in range(1, args.epoch):
        if epoch_num > lr_schedule[lr_sc_idx]:
            lr_sc_idx += 1
        epoch_lr = min_lr + (initial_lr - min_lr) * 0.5 * (1+np.cos((epoch_num-lr_schedule[lr_sc_idx-1])/(lr_schedule[lr_sc_idx] - lr_schedule[lr_sc_idx-1])*np.pi))
        # epoch_lr = initial_lr
        print("\n\nEpoch: {} Learning rate: {}".format(epoch_num, epoch_lr))
        print("Last epoch loss: {}, accuracy: {}".format(np.round(last_epoch_val_loss, 3), np.round(last_epoch_val_acc, 3)))
        
        if selected_c_num > 0:
            # get local models
            global_para, local_paras_round, train_loss_round = training_recorder.get_info(selected_clients)
            for ci in range(worker_num):
                agg_weight[ci] = agg_weight[ci] * beta

            for sci, ci in enumerate(selected_clients):
                if epoch_num == 1:
                    train_loss_mv[ci] = train_loss_round[sci]
                    client_local_paras.append(local_paras_round[sci])
                else:
                    train_loss_mv[ci] = beta * train_loss_mv[ci] + (1-beta) * train_loss_round[sci]
                    client_local_paras[ci] = beta * client_local_paras[ci] + (1-beta) * local_paras_round[sci]
                    agg_weight[ci] = 1.0 / worker_num

            # aggregation
            with torch.no_grad():
                is_first = True
                num_of_clients = len(selected_clients)
                for client_idx in range(num_of_clients):
                    if is_first:
                        global_para_round = local_paras_round[client_idx].clone().detach()
                        is_first = False
                    else:
                        global_para_round = global_para_round + local_paras_round[client_idx].clone().detach()
                global_para_round = global_para_round / num_of_clients
                torch.nn.utils.vector_to_parameters(global_para_round, global_model.parameters())

                if args.moving_global:
                    is_first = True
                    agg_weight = p_client.copy()
                    agg_weight_norm = agg_weight / np.sum(agg_weight)
                    print("*\naggregation weight", np.round(agg_weight_norm, 2))
                    for client_idx in range(worker_num):
                        if is_first:
                            global_para_round_1 = client_local_paras[client_idx].clone().detach() * agg_weight_norm[client_idx]
                            is_first = False
                        else:
                            global_para_round_1 = global_para_round_1 + client_local_paras[client_idx].clone().detach() * agg_weight_norm[client_idx]
                    torch.nn.utils.vector_to_parameters(global_para_round_1, teacher_model.parameters())
                else:
                    torch.nn.utils.vector_to_parameters(global_para_round, teacher_model.parameters())

        training_recorder.recorder.add_scalar('Resource', len(selected_clients), epoch_num)

        test_loss, acc = test(teacher_model, test_loader)
        print("Teacher model test -- loss: {}, accuracy: {}".format(np.round(test_loss, 3), np.round(acc, 3)))
        training_recorder.recorder.add_scalar('TestAccuracy-teacher', acc, epoch_num)
        training_recorder.recorder.add_scalar('TestLoss-teacher', test_loss, epoch_num)

        val_loss_1, val_acc_1 = test(global_model, validation_loader)
        reward_1 = val_acc_1-last_epoch_val_acc
        print("Reward of aggregation: {}\n**".format(reward_1))
        if args.client_frac <= 0 and epoch_num > args.start_select:
            print("**")
            agent_client.add_data(client_frac_idx, reward_1)
            print("**")
        print("Validation of aggregation: loss_1:{}, acc_1:{}".format(np.round(val_loss_1, 3), np.round(val_acc_1, 3)))
        training_recorder.recorder.add_scalar('ValAccuracy-agg', val_acc_1, epoch_num)
        training_recorder.recorder.add_scalar('ValLoss-agg', val_loss_1, epoch_num)
        # if epoch_num-lr_schedule[lr_sc_idx-1] > 5:
        #     if val_acc_1 - last_epoch_val_acc < -0.02:
        #         print("-*-resume 1-*-")
        #         torch.nn.utils.vector_to_parameters(global_para_tmp, global_model.parameters())
        #         val_loss_1, val_acc_1 = last_epoch_val_loss, last_epoch_val_acc
        
        test_loss, acc = test(global_model, test_loader)
        print("Global model test -- loss: {}, accuracy: {}".format(np.round(test_loss, 3), np.round(acc, 3)))
        training_recorder.recorder.add_scalar('TestAccuracy-global', acc, epoch_num)
        training_recorder.recorder.add_scalar('TestLoss-global', test_loss, epoch_num)

        if epoch_num > args.start_label:
            # produce pseudo labels
            time_1 = time.time()
            print()
            prediction_results, t_acc = get_results_val(train_loader, global_model)
            time_2 = time.time()
            print("Pseudo labeling time ", time_2 - time_1)
            training_recorder.recorder.add_scalar('TeacherAccuracy', t_acc, epoch_num)
            # select samples by thd
            avg_labels, selected_indices = select_unlabelled_data(prediction_results, sample_thd)
            time_3 = time.time()
            print("Pseudo label selection time ", time_3 - time_2)
            
            # retrain
            print("**")
            if len(selected_indices) >= args.tf_batch_size:
                global_para_backup = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
                optimizer = optim.SGD(global_model.parameters(), lr=epoch_lr, momentum=0.9, weight_decay=args.weight_decay)
                tf_train_loss, total_num, right_num, train_acc = train_on_pseduo(global_model, train_dataset, np.array(train_data_partition.use(worker_num)), selected_indices, avg_labels, optimizer, args.tf_batch_size)
                time_4 = time.time()
                print("Pseudo train time ", time_4 - time_3)
                val_loss_2, val_acc_2 = test(global_model, validation_loader)
                time_5 = time.time()
                print("validation time ", time_5 - time_4)
                reward_2 = val_acc_2 - val_acc_1
                if args.thd_avg <= 0:
                    agent_sample.add_data(sample_thd_idx, reward_2)
                print("Validation of retrain: loss_2:{}, acc_2:{}".format(np.round(val_loss_2, 3), np.round(val_acc_2, 3)))
                print("Reward of pseudo labels: {}\n**".format(reward_2))
                # if val_acc_2 - val_acc_1 < -0.02 or val_acc_2 < 0.12:
                #     print("-*-resume 2-*-")
                #     torch.nn.utils.vector_to_parameters(global_para_backup, global_model.parameters())
                #     val_loss_2, val_acc_2 = val_loss_1, val_acc_1
            else:
                val_loss_2, val_acc_2 = val_loss_1, val_acc_1

            if len(selected_indices) >= args.tf_batch_size:
                training_recorder.recorder.add_scalar('Selected-num', total_num, epoch_num)
                training_recorder.recorder.add_scalar('Right-num', right_num, epoch_num)
                training_recorder.recorder.add_scalar('TfLoss', tf_train_loss, epoch_num)
                training_recorder.recorder.add_scalar('TrainAcc-Pseudo', train_acc, epoch_num)

            training_recorder.recorder.add_scalar('ValAccuracy-server', val_acc_2, epoch_num)
            training_recorder.recorder.add_scalar('ValLoss-server', val_loss_2, epoch_num)

        if epoch_num == args.start_label:
            val_loss_2, val_acc_2 = test(global_model, validation_loader)

        if epoch_num >= args.start_select:
            if epoch_num % 3 == 0:
                if args.client_frac > 0:
                    client_frac = args.client_frac
                else:
                    client_frac_idx = agent_client.select()
                    client_frac = client_frac_table[client_frac_idx]
            print("Fraction of client:{}".format(client_frac))

            # client sampling
            if args.importance:
                loss_e = np.exp(train_loss_mv * hyperpara[2])
                p_client = loss_e / np.sum(loss_e)
            else:
                p_client = np.ones((worker_num,)) / worker_num
            # print("probability of clients", np.round(p_client, 3))
            selected_clients = np.random.choice(worker_num, int(np.ceil(worker_num*client_frac)), False, p_client)
        else:
            selected_clients = list(range(worker_num))
        last_epoch_val_acc = val_acc_1
        last_epoch_val_loss = val_loss_1

        if epoch_num >= args.start_label:
            if args.thd_avg > 0:
                sample_thd = args.thd_avg
            else:
                sample_thd_idx = agent_sample.select()
                sample_thd = sample_thd_table[sample_thd_idx]
            print("**\nThreshold of sample:{}".format(sample_thd))

            last_epoch_val_acc = val_acc_2
            last_epoch_val_loss = val_loss_2
        
        print("Selected clients: ", selected_clients)
        selected_c_num = len(selected_clients)
        info_worker_list = list()
        if selected_c_num > 0:
            for ci in selected_clients:
                info_worker_list.append(common_config.worker_list[ci])
            global_para_tmp = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
            communication_parallel(info_worker_list, action="send", data=(global_para_tmp, epoch_num))

    communication_parallel(common_config.worker_list, action="send", data=(0, -1))
    # close socket
    for worker in common_config.worker_list:
        worker.socket.shutdown(2)
        worker.socket.close()

class TrainingRecorder(object):
    def __init__(self, worker_list, recorder):
        self.worker_list = worker_list
        self.worker_num = len(worker_list)
        self.epoch = 0
        self.recorder = recorder
        self.local_paras = list()

    def get_info(self, client_idxes):
        self.epoch += 1
        info_worker_list = list()
        for ci in client_idxes:
            info_worker_list.append(self.worker_list[ci])
        communication_parallel(info_worker_list, action="get")
        avg_acc = 0.0
        avg_test_loss = 0.0
        avg_train_loss = 0.0

        self.local_paras = list()
        train_loss_list = list()
        for worker in info_worker_list:
            _, acc, train_loss, test_loss, local_para = worker.train_info[-1]
            train_loss_list.append(train_loss)
            self.local_paras.append(local_para)
            self.recorder.add_scalar('Accuracy/worker_' + str(worker.idx), acc, self.epoch)
            self.recorder.add_scalar('Test_loss/worker_' + str(worker.idx), test_loss, self.epoch)
            self.recorder.add_scalar('Train_loss/worker_' + str(worker.idx), train_loss, self.epoch)

            avg_acc += acc
            avg_test_loss += test_loss
            avg_train_loss += train_loss
        
        worker_num = len(client_idxes)
        global_para = self.local_paras[0].clone().detach()
        for worker_idx in range(worker_num):
            global_para = global_para + self.local_paras[worker_idx].clone().detach()
        global_para = global_para / worker_num

        avg_acc /= worker_num
        avg_test_loss /= worker_num
        avg_train_loss /= worker_num
        
        self.recorder.add_scalar('Accuracy/average', avg_acc, self.epoch)
        self.recorder.add_scalar('Test_loss/average', avg_test_loss, self.epoch)
        self.recorder.add_scalar('Train_loss/average', avg_train_loss, self.epoch)
        print("Epoch: {}, average accuracy: {}, average test loss: {}, average train loss: {}".format(self.epoch, np.round(avg_acc, 3), np.round(avg_test_loss, 3), np.round(avg_train_loss, 3)))

        return global_para, self.local_paras, train_loss_list

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get":
                tasks.append(loop.run_in_executor(executor, worker.get_config))
            elif action == "send":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)

def non_iid_partition_10(server_ratio, main_ratio, validation_ratio=0.1):
    worker_num = 10
    partition_sizes = np.ones((10, worker_num+2)) * (1 - server_ratio - main_ratio - validation_ratio) / 9

    for class_idx in range(worker_num):
        partition_sizes[class_idx][10] = server_ratio
        partition_sizes[class_idx][11] = validation_ratio
        partition_sizes[class_idx][class_idx] = main_ratio

    return partition_sizes

def non_iid_partition_20(server_ratio, main_ratio, validation_ratio=0.1):
    partition_sizes = np.ones((10, 22)) * ((1 - server_ratio - main_ratio - validation_ratio) / 18)

    for class_idx in range(10):
        partition_sizes[class_idx][20] = server_ratio
        partition_sizes[class_idx][21] = validation_ratio
        partition_sizes[class_idx][class_idx*2] = main_ratio / 2
        partition_sizes[class_idx][class_idx*2+1] = main_ratio / 2

    return partition_sizes

def non_iid_partition_25(server_ratio, main_ratio, validation_ratio=0.1):
    partition_sizes = np.zeros((10, 27))
    worker_idx = 0
    for class_idx in range(10):
        class2client_num = 2 + class_idx % 2
        partition_sizes[class_idx] = np.ones((27, )) * ((1 - server_ratio - main_ratio - validation_ratio) / (30-class2client_num))
        partition_sizes[class_idx][25] = server_ratio
        partition_sizes[class_idx][26] = validation_ratio

        for shift_idx in range(class2client_num):
            partition_sizes[class_idx][worker_idx] = main_ratio / class2client_num
            worker_idx += 1

    return partition_sizes

def non_iid_partition_30(server_ratio, main_ratio, validation_ratio=0.1):
    partition_sizes = np.ones((10, 32)) * ((1 - server_ratio - main_ratio - validation_ratio) / 27)

    for class_idx in range(10):
        partition_sizes[class_idx][30] = server_ratio
        partition_sizes[class_idx][31] = validation_ratio
        partition_sizes[class_idx][class_idx*3] = main_ratio / 3
        partition_sizes[class_idx][class_idx*3+1] = main_ratio / 3
        partition_sizes[class_idx][class_idx*3+2] = main_ratio / 3

    return partition_sizes

def partition_data(dataset_type, server_ratio, main_ratio, validation_ratio=0.1, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR100":
        data_pattern = 0
        test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((100, worker_num)) * (1 / (worker_num-data_pattern))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx
            for _ in range(data_pattern):
                partition_sizes[tmp_idx*worker_num:(tmp_idx+1)*worker_num, worker_idx] = 0
                tmp_idx = (tmp_idx + 1) % 10
    elif dataset_type == "CIFAR10":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
        if worker_num == 10:
            partition_sizes = non_iid_partition_10(server_ratio, main_ratio, validation_ratio)
        elif worker_num == 20:
            partition_sizes = non_iid_partition_20(server_ratio, main_ratio, validation_ratio)
        elif worker_num == 25:
            partition_sizes = non_iid_partition_25(server_ratio, main_ratio, validation_ratio)
        elif worker_num == 30:
            partition_sizes = non_iid_partition_30(server_ratio, main_ratio, validation_ratio)
        
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)
 
    return train_data_partition, test_data_partition

def test(model, data_loader, device=torch.device("cuda")):
    model.eval()
    model = model.to(device)
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    return test_loss, test_accuracy

def get_results_val(data_loader, label_model, device=torch.device("cuda")):
    setResults = list()
    label_model.eval()
    data_loader = data_loader.loader

    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = label_model(data)
            softmax1 = F.softmax(output, dim=1).cpu().detach().numpy()
            avg_pred = softmax1.copy()
            setResults.extend(softmax1.copy())

            pred = avg_pred.argmax(1)
            correct = correct + np.sum(pred == target.cpu().detach().numpy().reshape(pred.shape))

    t_acc = correct/len(data_loader.dataset)
    print("teachers' acc in val: {}".format(t_acc))
    return np.array(setResults), t_acc

def select_unlabelled_data(prediction_results, p_th):
    avg_labels = prediction_results
    max_label = np.argmax(avg_labels, axis=1)

    selected_indices = list()
    for data_idx in range(len(prediction_results)):
        if avg_labels[data_idx][max_label[data_idx]] >= p_th:
            selected_indices.append(data_idx)

    print("num of selected samples: ", len(selected_indices))
    return np.array(avg_labels)[selected_indices], np.array(selected_indices)

def count_dataset(loader, soft_labels, tf_batch_size):
    counts = np.zeros(len(loader.loader.dataset.classes))
    right = np.zeros(len(loader.loader.dataset.classes))

    st_matrix = np.zeros((len(loader.loader.dataset.classes), len(loader.loader.dataset.classes)))
    for data_idx, (_, target) in enumerate(loader.loader):
        predu = torch.from_numpy(soft_labels[data_idx*tf_batch_size:(data_idx+1)*tf_batch_size])
        predu = predu.argmax(1)
        batch_correct = predu.eq(target.view_as(predu))

        labels = target.view(-1).numpy()
        predu = predu.view(-1).numpy()
        for label_idx, label in enumerate(labels):
            counts[label] += 1
            st_matrix[label][predu[label_idx]] += 1
            if batch_correct[label_idx] == True:
                right[label] += 1
    print(st_matrix.astype(np.int))
    print("class counts: ", counts.astype(np.int))
    print("total data count: ", np.sum(counts))
    print("right class counts: ", right.astype(np.int))
    print("total right data count: ", np.sum(right))

    return np.sum(counts), np.sum(right)

def train_on_pseduo(model, train_dataset, unlabelled_indices, selected_indices, soft_labels, optimizer, tf_batch_size, device=torch.device("cuda")):
    if len(selected_indices) <= 0:
        return
    model.train()
    model = model.to(device)
    selected_shuffle = [i for i in range(len(selected_indices))]
    np.random.shuffle(selected_shuffle)
    selected_indices = selected_indices[selected_shuffle]
    soft_labels = soft_labels[selected_shuffle]

    train_loader = datasets.create_dataloaders(train_dataset, batch_size=tf_batch_size, selected_idxs=unlabelled_indices[selected_indices], shuffle=False)
    total_num, right_num = count_dataset(train_loader, soft_labels, tf_batch_size)

    data_loader = train_loader.loader
    samples_num = 0

    train_loss = 0.0
    correct = 0

    correct_s = 0
    
    for data_idx, (data, label) in enumerate(data_loader):

        target = torch.from_numpy(soft_labels[data_idx*tf_batch_size:(data_idx+1)*tf_batch_size])
        data, target, label = data.to(device), target.to(device), label.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        
        loss = F.cross_entropy(output, target.argmax(1))
        # loss = F.cross_entropy(output, label)

        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

        pred = target.argmax(1, keepdim=True)
        batch_correct = pred.eq(label.view_as(pred)).sum().item()
        correct += batch_correct

        pred_s = output.argmax(1, keepdim=True)
        batch_correct_s = pred.eq(pred_s).sum().item()
        correct_s += batch_correct_s

    if samples_num != 0:
        train_loss /= samples_num
        # print("sample num: ", samples_num)
        test_accuracy = np.float(1.0 * correct / samples_num)
        print("teacher's acc : {}".format(test_accuracy))
        test_accuracy = np.float(1.0 * correct_s / samples_num)
        print("student's training acc : {}".format(test_accuracy))
    
    return train_loss, total_num, right_num, test_accuracy

def count_dataset_total(loader):
    counts = np.zeros(len(loader.loader.dataset.classes))
    for _, target in loader.loader:
        labels = target.view(-1).numpy()
        for label in labels:
            counts[label] += 1
    print("class counts: ", counts)
    print("total data count: ", np.sum(counts))

def killport(port):
    command = '''kill -9 $(netstat -nlp | grep :''' + str(
        port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
    os.system(command)

if __name__ == "__main__":
    main()
