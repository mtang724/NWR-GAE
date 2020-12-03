import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        # P = F.softmax(inputs)
        P = inputs

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class HungarianLoss():
    def __init__(self, predictions, targets, mask, pool):
        self.predictions = predictions
        self.targets = targets
        self.mask = mask
        self.pool = pool

    def outer(self, a, b=None):
        if b is None:
            b = a
        size_a = tuple(a.size()) + (b.size()[-1],)
        size_b = tuple(b.size()) + (a.size()[-1],)
        a = a.unsqueeze(dim=-1).expand(*size_a)
        b = b.unsqueeze(dim=-2).expand(*size_b)
        return a, b

    def per_sample_hungarian_loss(self, sample_np):
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
        return row_idx, col_idx

    def hungarian_loss(self):
        # predictions and targets shape :: (n, c, s)
        predictions = self.predictions[:self.mask,:]
        targets = self.targets[:self.mask,:]
        predictions = predictions.permute(0, 2, 1)
        targets = targets.permute(0, 2, 1)
        predictions, targets = self.outer(predictions, targets)
        # squared_error shape :: (n, s, s)
        squared_error = (predictions - targets).pow(2).mean(1)

        squared_error_np = squared_error.detach().cpu().numpy()
        indices = self.pool.map(self.per_sample_hungarian_loss, squared_error_np)
        losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
        total_loss = torch.mean(torch.stack(list(losses)))
        return total_loss
