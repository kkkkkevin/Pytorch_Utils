import numpy as np
import torch
import util.box_ops as box_ops


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def avg(self):
        return (self.sum / self.count) if self.count > 0 else 0


@torch.no_grad()
def avg_precision(logit, pboxes, tboxes, reduce=True):
    idx = logit.gt(0)
    if sum(idx) == 0 and len(tboxes) == 0:
        return 1 if reduce else [1] * 6
    if sum(idx) > 0 and len(tboxes) == 0:
        return 0 if reduce else [0] * 6

    pboxes = pboxes[idx]
    logit = logit[idx]

    idx = logit.argsort(descending=True)
    pboxes = box_ops.box_cxcywh_to_xyxy(pboxes.detach()[idx])
    tboxes = box_ops.box_cxcywh_to_xyxy(tboxes)

    iou = box_ops.box_iou(pboxes, tboxes)[0].cpu().numpy()
    prec = [precision(iou, th) for th in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]]
    if reduce:
        return sum(prec) / 6
    return prec


def precision(iou, th):
    # if iou.shape==(0,0): return 1
    # if min(*iou.shape)==0: return 0
    tp = 0
    iou = iou.copy()
    num_pred, num_gt = iou.shape
    for i in range(num_pred):
        _iou = iou[i]
        n_hits = (_iou > th).sum()
        if n_hits > 0:
            tp += 1
            j = np.argmax(_iou)
            iou[:, j] = 0
    return tp / (num_pred + num_gt - tp)


def challenge_metric(outputs, targets):
    logits = outputs['pred_logits']
    boxes = outputs['pred_boxes']
    return sum(avg_precision(logit[:, 0] - logit[:, 1], box, target['boxes'])
               for logit, box, target in zip(logits, boxes, targets)) / len(logits)

    return {target['image_id']: avg_precision(
        logit[:, 0] - logit[:, 1], box, target['boxes']) for logit, box, target in zip(logits, boxes, targets)}
