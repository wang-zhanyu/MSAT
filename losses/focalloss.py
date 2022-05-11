import torch
import torch.nn.functional as F

class FocalLoss:
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def __call__(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        pt = torch.exp(-F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')).detach()
        sample_weight = (1 - pt) **  self.gamma
        return F.binary_cross_entropy_with_logits(y_pred, y_true, weight=sample_weight, pos_weight=self.alpha_t)


if __name__ == '__main__':
    outputs = torch.tensor([[2, 1.],
                            [2.5, 1]], device='cuda')
    targets = torch.tensor([0, 1], device='cuda')
    print(torch.nn.functional.softmax(outputs, dim=1))

    fl= FocalLoss([0.5, 0.5], 2)

    print(fl(outputs, targets))