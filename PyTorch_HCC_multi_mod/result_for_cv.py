from torch.autograd import Variable
import time
from utils import OsJoin
from utils import AverageMeter,calculate_accuracy,calculate_recall
import torch
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TestSet, TrainSet, ValidSet
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
        data_time.update(time.time() - end_time)
        labels = list(map(int, labels))
        label_true += labels
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.type(torch.FloatTensor)

        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            labels = Variable(labels)
            outputs = model(inputs)
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
opt.batch_size = 64
use_focal_loss = False

label_predicts = []
label_trues = []
epoch_num = 20

for fold in range(1, 6):
    print('*'*50)
    if fold == 2:
        print('fold', fold)
        resume_path = '/home/amax/Wendy/zhongzhong_result/net_model_fold%s.pkl'% (str(fold))
        # print('resume_file', resume_path)
        # opt.resume_path = OsJoin(opt.root_path, opt.resume_path)
        # test_data = TestSet()
        test_data = TrainSet(fold_id=2)
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
        print('loading checkpoint{}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        model = torch.load(resume_path)

        label_predict_pro, label_true = test_epoch(epoch_num, test_loader, model, criterion, opt, test_logger)
        # print('label_predict_pro', label_predict_pro)
        # print('label_ture', label_true)

        label_predict = [i[1] for i in label_predict_pro]
        for i in range(len(label_true)):
            label_trues.append(label_true[i])
            label_predicts.append(label_predict[i])
        print('Fold %s result' % (fold))
        fpr, tpr, thresholds = roc_curve(label_true, label_predict)
        label_predict = [1 if i >= 0.5 else 0 for i in label_predict]
        print('label_pred', label_predict)
        print('auc', auc(fpr, tpr))
        print('accuracy_score', accuracy_score(label_true, label_predict))
        print(classification_report(label_true, label_predict))


print('Final result')
fpr, tpr, thresholds = roc_curve(label_trues, label_predicts)
label_predicts = [1 if i >= 0.5 else 0 for i in label_predicts]
print('label_pred', label_predicts)
print('auc', auc(fpr, tpr))
print('accuracy_score', accuracy_score(label_trues, label_predicts))
print(classification_report(label_trues, label_predicts))
