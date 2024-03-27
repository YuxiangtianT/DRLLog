# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=28, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ * ce_loss(xi,yi)
        focal_loss损失函数 详细实现步骤
        :param alpha:   阿尔法α,类别权重.
        当α是列表时,为各类别权重；当α为常数时,类别权重为[α, 1-α, 1-α, ....]
        常用于目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average

        # 设置类别权重 alpha
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            # α可以以list方式输入,size:[num_classes]用于对不同类别精细地赋予权重
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):

        """
        focal_loss损失计算
        分别对应与检测与分类任务, B批次, N检测框数, C类别数
        :param preds:   预测类别. size:[B,N,C] or [B,C]
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        # 将预测概率展平，以适应计算
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)

        # 应用 log_softmax 和 softmax
        preds_log_softmax = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_log_softmax)

        # 选择真实标签对应的概率
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_log_softmax.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1 - preds_softmax)为focal loss中(1-pt)**γ
        # 计算 focal loss
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        # 用 alpha 进行加权
        loss = torch.mul(alpha, loss.t())

        # 平均或求和损失
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
