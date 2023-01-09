import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy, calculate_recall, calculate_F1
import numpy as np
from sklearn import metrics


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def val_epoch(epoch, data_loader, model, criterion, opt, logger, writer):
    print('validation at epoch {}'.format(epoch) )
    # print('criterion', criterion)
    model.eval()

    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    # recalls = AverageMeter()
    # F1s = AverageMeter()

    writer = writer

    end_time = time.time()
    running_corrects = 0
    size_data = 0
    preds_all = []
    labels_all = []
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
        _, preds = torch.max(outputs.data, 1)
        for idx in range(len(preds)):
            preds_all.append(preds[idx].cpu().numpy())
            labels_all.append(labels[idx].cpu().numpy())
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

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            acc=accuracies))
    epoch_acc = running_corrects.cpu().numpy() / size_data
    # print('labels_all', labels_all)
    fpr, tpr, thresholds = metrics.roc_curve(labels_all, preds_all, pos_label=1)
    recall_score = metrics.recall_score(labels_all, preds_all)
    F1_score = metrics.f1_score(labels_all, preds_all)
    epoch_auc = metrics.auc(fpr, tpr)
    conf_matrix = torch.zeros(2, 2)
    conf_matrix = confusion_matrix(preds_all, labels=labels_all, conf_matrix=conf_matrix)
    logger.log({'epoch': epoch, 'loss': round(losses.avg.item(), 4),
                'acc': round(accuracies.avg.item(), 4), 'recall': recall_score})
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/accuracy', accuracies.avg, epoch)

    return losses.avg, round(epoch_acc, 3), conf_matrix, round(epoch_auc, 3), round(recall_score, 3), round(F1_score, 3)
