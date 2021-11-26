import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CrossEntropy2D(nn.Layer):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2D, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of shape "nclasses"
        """
        assert target.stop_gradient
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.shape[0] == target.shape[0], "{0} vs {1} ".format(predict.shape[0], target.shape(0))
        assert predict.shape[2] == target.shape[1], "{0} vs {1} ".format(predict.shape[2], target.shape[1])
        assert predict.shape[3] == target.shape[2], "{0} vs {1} ".format(predict.shape[3], target.shape[3])
        n, c, h, w = predict.shape
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.dim():
            return paddle.zeros(1)
        predict = predict.transpose([0, 2, 3, 1])
        predict = predict[target_mask.reshape([n, h, w, 1]).tile([1, 1, 1, c])].reshape([-1, c])
        loss = F.cross_entropy(predict, target, weight=weight)  # todo modified delete size_averge
        return loss