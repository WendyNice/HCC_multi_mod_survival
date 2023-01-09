from torch.autograd import Variable
import time
from utils import OsJoin
from utils import AverageMeter,calculate_accuracy,calculate_recall
import torch
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TestSet, ValidSet, TrainSet, AllSet, ExtraValidSet, ExtraAllValidSet
from utils import Logger
from torch import nn
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, auc, roc_curve
from focal_loss import FocalLoss
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import os
import numpy as np


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(cm, classes, accuracy, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=0))
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        if cm[i, j] > thresh:
            plt.text(j, i, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white")
        else:
            plt.text(j, i, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="black")
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()


def test_epoch(data_loader, model, criterion, opt):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    # recalls = AverageMeter()
    # F1s = AverageMeter()


    end_time = time.time()
    running_corrects = 0
    size_data = 0
    preds_all = []
    labels_all = []
    output_all = []
    for i, (inputs, labels) in enumerate(data_loader):
        # print('size', inputs.size())
        data_time.update(time.time() - end_time)
        labels = list(map(int, labels))
        # inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.type(torch.FloatTensor)

        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda()
        else:
            labels = torch.LongTensor(labels)
        # with torch.no_grad():
        inputs = Variable(inputs)
        labels = Variable(labels)
        outputs = model(inputs)
        outputs_ls = list(outputs.cpu().detach().numpy())
        print('outputs_ls', outputs_ls)

        _, preds = torch.max(outputs.data, 1)
        for idx in range(len(preds)):
            preds_all.append(preds[idx].cpu().numpy())
            labels_all.append(labels[idx].cpu().numpy())
            output_all.append(outputs_ls[idx])
        # print('outputs', outputs.shape, labels.shape)
        loss = criterion(outputs, labels)

        running_corrects += torch.sum(preds == labels.data)
        size_data += len(labels)
        acc = calculate_accuracy(outputs, labels)
        # recall = calculate_recall(outputs, labels)
        # F1 = calculate_F1(outputs, labels)
        # print('outputs', outputs)
        # print('labels', labels)
        losses.update(loss.data, inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        # recalls.update(recall, inputs.size(0))
        # F1s.update(F1, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            acc=accuracies))
    epoch_acc = running_corrects.cpu().numpy() / size_data
    print('labels_all', labels_all)
    print('preds_all', preds_all)
    fpr, tpr, thresholds = metrics.roc_curve(labels_all, preds_all, pos_label=1)
    recall_score = metrics.recall_score(labels_all, preds_all)
    F1_score = metrics.f1_score(labels_all, preds_all)
    epoch_auc = metrics.auc(fpr, tpr)
    conf_matrix = torch.zeros(2, 2)
    conf_matrix = confusion_matrix(preds_all, labels=labels_all, conf_matrix=conf_matrix)
    return losses.avg, round(epoch_acc, 3), conf_matrix, round(epoch_auc, 3), round(recall_score, 3), \
           round(F1_score, 3), output_all


opt = parse_opts()
use_focal_loss = True
opt.no_cuda = True
print('opt.no_cuda', opt.no_cuda)
if opt.test:
    opt.batch_size = 64
    print('opt.batch_size', opt.batch_size)

    # task = 'OS'
    # file_name = 'net_model_0.669.pkl'
    # # file_name = 'net_model_0.595.pkl'
    # opt.resume_path = '/data/Wendy/HCC/MR_clf/result/output_' + task + '_resample_100_shape/' + file_name
    # save_path = '/data/Wendy/HCC/MR_clf/result/output_' + task + '_resample_100_shape/feature_dl'

    task = 'RFS'
    file_name = 'net_model_0.622.pkl'
    opt.resume_path = '/data/Wendy/HCC/MR_clf/result/output_' + task + '_resample_100_shape/' + file_name
    save_path = '/data/Wendy/HCC/MR_clf/result/output_' + task + '_resample_100_shape/feature_dl'

    #
    # test_data = ValidSet(fold_id='_' + task + '_resample')
    # save_file = os.path.join(save_path, 'valid.npy')
    #
    # test_data = TrainSet(fold_id='_' + task + '_resample')
    # save_file = os.path.join(save_path, 'train.npy')
    #
    # test_data = AllSet(fold_id='_' + task + '_resample')
    # save_file = os.path.join(save_path, 'all.npy')
    #
    # test_data = ExtraValidSet(fold_id='_' + task + '_resample')
    # save_file = os.path.join(save_path, 'extra_valid.npy')

    test_data = ExtraAllValidSet(fold_id='_' + task + '_resample')
    save_file = os.path.join(save_path, 'extra_all_valid.npy')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if opt.resume_path:
        opt.resume_path = OsJoin(opt.root_path, opt.resume_path)
    print('opt.resume_path', opt.resume_path)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False,
                                                        num_workers =0, pin_memory=False)
    model, parameters = generate_model(opt)
    if use_focal_loss:
        criterion = FocalLoss(gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()

    if not opt.no_cuda:
        criterion = criterion.cuda()
    log_path = OsJoin(opt.result_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth))
    test_logger = Logger(
        OsJoin(log_path, 'test1.log'), ['epoch', 'loss', 'acc', 'recall'])
    print('loading checkpoint{}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    model = torch.load(opt.resume_path)

    validation_loss, epoch_acc, conf_matrix, epoch_auc, recall_score, F1_score, output = test_epoch(test_loader, model,
                                                                                           criterion, opt)
    print('epoch_auc', epoch_auc, recall_score, F1_score)
    best_auc = epoch_auc
    plot_confusion_matrix(conf_matrix.numpy(), accuracy=epoch_auc, classes=['0', '1'], normalize=False,
                                          title='Confusion_matrix')
    output = np.array(output)
    print('output', output.shape)
    np.save(save_file, output)


'''
RFS
    epoch_auc
    0.57
    0.717
    0.55
    Confusion
    matrix, without
    normalization
    [[30. 13.]
     [41. 33.]]
    output(117, 2)

OS
epoch_auc 0.603 0.522 0.545
Confusion matrix, without normalization
[[39. 22.]
 [18. 24.]]
output (103, 2)

'''





























# def calculate_test_results(output_buffer,sample_id,test_results,labels):
#     outputs =torch.stack(output_buffer)
#     average_score = torch.mean(outputs,dim=0)
#     sorted_scores,locs = torch.topk(average_score,k=1)
#     results=[]
#     for i in range(sorted_scores.size(0)):
#         score = copy.deepcopy(sorted_scores[i])
#         if isinstance(score, torch.Tensor):
#             score = score.data.cpu().numpy()
#             score = score.item()
#         results.append({
#             'label':labels[i],
#             'score':score
#         })
#     test_results['results'][sample_id] = results
#
# def test(data_loader, model, opt, labels):
#     print('test')
#
#     model.eval()
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#
#     end_time = time.time()
#     output_buffer = []
#     sample_id = ''
#     test_results = {'results': {}}
#     with torch.no_grad():
#         for i, (inputs, targets) in enumerate(data_loader):
#             data_time.update(time.time() - end_time)
#
#             inputs = torch.unsqueeze(inputs, 1)  # 在 1 的位置加一个维度
#             inputs = Variable(inputs)
#             outputs = model(inputs)
#             # if not opt.no_softmax_in_test:
#             #outputs = F.softmax(outputs)
#
#             for j in range(outputs.size(0)):
#                 if not (i == 0 and j == 0):
#                     calculate_test_results(output_buffer, sample_id, test_results, labels)
#                     output_buffer = []
#                 output_buffer.append(outputs[j].data.cpu())
#                 sample_id = labels[j]
#             if (i % 100) == 0:
#                 with open(
#                         OsJoin(opt.result_path, '{}.json'.format(
#                             opt.test_subset)), 'w') as f:
#                     json.dump(test_results, f)
#
#             batch_time.update(time.time() - end_time)
#             end_time = time.time()
#
#             print('[{}/{}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
#                       i + 1,
#                       len(data_loader),
#                       batch_time=batch_time,
#                       data_time=data_time))
#     with open(
#             OsJoin(opt.result_path, opt.data_type, opt.model_name, str(opt.model_depth), '{}.json'.format(opt.test_subset)),
#             'w') as f:
#         json.dump(test_results, f)
