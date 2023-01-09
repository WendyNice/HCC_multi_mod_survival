import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss_1(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss_1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            print('input', input)
            print('target', target)
            input = input.view(input.size(0), input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class FocalLoss_2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        super(FocalLoss_2, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, (float, int)):
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # print('inputs', inputs)
        # print('targets', targets)
        alpha = self.alpha
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        # print('P', P)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = class_mask.requires_grad_()
        ids = targets.view(-1, 1)
        class_mask.data.scatter_(1, ids.data, 1.)
        # print('class_mask', class_mask)
        probs = (P * class_mask).sum(1).view(-1, 1)  # 预测对的概率
        print('probs', probs)
        log_p = probs.log()
        # loss = torch.pow((1 - probs), self.gamma) * log_p
        # print('alpha', alpha)
        # print('loss.t()', loss)
        # batch_loss = -alpha * loss.t()

        device = torch.device('cuda:0')  # 假如我使用的GPU为cuda:0
        alpha = alpha.to(device)
        probs = probs.to(device)
        log_p = log_p.to(device)
        print('log_p', log_p)
        print('alpha', alpha)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        print('batch_loss', batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
            print('loss', loss)
        else:
            loss = batch_loss.sum()
        # print('loss: ', loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """

        # assert preds.dim()==2 and labels.dim()==1
        print('preds', preds)
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax
        # print('labels', labels.dtype, labels)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = alpha.gather(0, labels.view(-1))
        # print('alpha', alpha)
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        # print('labels', labels)

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
