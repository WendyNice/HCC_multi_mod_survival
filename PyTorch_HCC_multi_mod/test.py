from torch.autograd import Variable
import time
from utils import OsJoin
from utils import AverageMeter,calculate_accuracy,calculate_recall
import torch
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TestSet
from utils import Logger
from torch import nn
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, auc, roc_curve
from focal_loss import FocalLoss


def test_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('test at epoch {}'.format(epoch))
    model.eval()

    label_true = []
    label_predict = []
    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    recalls = AverageMeter()

    end_time = time.time()
    for i, (inputs, labels) in enumerate(data_loader):
        print('shape', inputs.shape)
        print('labels', labels)
        data_time.update(time.time() - end_time)
        labels = list(map(int, labels))
        label_true += labels
        # inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.type(torch.FloatTensor)

        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            labels = Variable(labels)
            outputs = model(inputs)
            # print()
            # print('outputs.cuda().data.cpu().numpy()', outputs.cuda().data.cpu().numpy().shape,
            #       outputs.cuda().data.cpu().numpy())
            label_predict += [outputs.cuda().data.cpu().numpy()[0]]
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            recall = calculate_recall(outputs, labels)

        losses.update(loss.data, inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        recalls.update(recall, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t' 
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Recall {recall.val:.3f} ({recall.avg:.3f})'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            acc=accuracies,
            recall=recalls))

    logger.log({'epoch': epoch, 'loss': round(losses.avg.item(), 4), 'acc': round(accuracies.avg.item(), 4),
                                                                'recall': round(recalls.avg.item(), 4)})
    return label_predict, label_true


opt = parse_opts()
use_focal_loss = True
if opt.test:
    opt.batch_size = 1
    print('opt.batch_size', opt.batch_size)
    opt.resume_path = r'/home/amax/Wendy/SYSU-hcc/data_clf/3d_result/net_model_0.7719298245614035.pkl'
    if opt.resume_path:
        opt.resume_path = OsJoin(opt.root_path, opt.resume_path)
    print('opt.resume_path', opt.resume_path)
    test_data = TestSet()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
                                                        num_workers = 0, pin_memory=True)
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

    label_predict_pro, label_ture = test_epoch(150, test_loader, model, criterion, opt, test_logger)

    print('label_predict_pro', label_predict_pro)
    label_predict = [i[1] for i in label_predict_pro]
    print('label_predict', label_predict)
    fpr, tpr, thresholds = roc_curve(label_ture, label_predict)

    print('label_ture', label_ture)
    label_predict = [1 if i >= 0.5 else 0 for i in label_predict]
    print('label_pred', label_predict)
    print('auc', auc(fpr, tpr))
    print('accuracy_score', accuracy_score(label_ture, label_predict))
    print(classification_report(label_ture, label_predict))

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
