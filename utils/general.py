import paddle
import glob
import math
import os
import random
import shutil
import subprocess
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from sys import platform
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from utils.torch_utils import init_seeds, is_parallel

paddle.set_printoptions(linewidth=320, precision=5)
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})
matplotlib.rc("font", **{"size": 11})
cv2.setNumThreads(0)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        paddle.distributed.barrier()
    yield
    if local_rank == 0:
        paddle.distributed.barrier()


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_seeds(seed=seed)


def get_latest_run(search_dir="./runs"):
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime)


def check_git_status():
    if platform in ["linux", "darwin"] and not os.path.isfile("/.dockerenv"):
        s = subprocess.check_output(
            "if [ -d .git ]; then git fetch && git status -uno; fi", shell=True
        ).decode("utf-8")
        if "Your branch is behind" in s:
            print(s[s.find("Your branch is behind") : s.find("\n\n")] + "\n")


def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(
            "WARNING: --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    print("\nAnalyzing anchors... ", end="")
    m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = paddle.to_tensor(
        data=np.concatenate(
            [(l[:, 3:5] * s) for s, l in zip(shapes * scale, dataset.labels)]
        )
    ).astype(dtype="float32")

    def metric(k):
        r = wh[:, None] / k[None]
        x = paddle.minimum(r, 1.0 / r).minimum(2)[0]
        best = x.max(1)[0]
        aat = (x > 1.0 / thr).astype(dtype="float32").sum(axis=1).mean()
        bpr = (best > 1.0 / thr).astype(dtype="float32").mean()
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().reshape((-1, 2)))
    print(
        "anchors/target = %.2f, Best Possible Recall (BPR) = %.4f" % (aat, bpr), end=""
    )
    if bpr < 0.98:
        print(". Attempting to generate improved anchors, please wait..." % bpr)
        na = m.anchor_grid.numel() // 2
        new_anchors = kmean_anchors(
            dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False
        )
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:
            new_anchors = paddle.to_tensor(
                data=new_anchors, place=m.anchors.device
            ).astype(dtype=m.anchors.dtype)
            m.anchor_grid[:] = new_anchors.clone().view_as(other=m.anchor_grid)
            m.anchors[:] = new_anchors.clone().view_as(other=m.anchors) / m.stride.to(
                m.anchors.device
            ).reshape((-1, 1, 1))
            check_anchor_order(m)
            print(
                "New anchors saved to model. Update model *.yaml to use these anchors in the future."
            )
        else:
            print(
                "Original anchors better than new anchors. Proceeding with original anchors."
            )
    print("")


def check_anchor_order(m):
    a = m.anchor_grid.prod(-1).reshape((-1,))
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        print("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_file(file):
    if os.path.isfile(file) or file == "":
        return file
    else:
        files = glob.glob("./**/" + file, recursive=True)
        assert len(files), "File Not Found: %s" % file
        return files[0]


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:
        return paddle.to_tensor(data=[])
    labels = np.concatenate(labels, 0)
    classes = labels[:, 0].astype(np.int)
    weights = np.bincount(classes, minlength=nc)
    weights[weights == 0] = 1
    weights = 1 / weights
    weights /= weights.sum()
    return paddle.to_tensor(data=weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    n = len(labels)
    class_counts = np.array(
        [np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)]
    )
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(axis=1)
    return image_weights


def coco80_to_coco91_class():
    x = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    return x


def xyxy2xywh(x):
    y = paddle.zeros_like(x=x) if isinstance(x, paddle.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    y = paddle.zeros_like(x=x) if isinstance(x, paddle.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    boxes[:, 0].clip_(min=0, max=img_shape[1])
    boxes[:, 1].clip_(min=0, max=img_shape[0])
    boxes[:, 2].clip_(min=0, max=img_shape[1])
    boxes[:, 3].clip_(min=0, max=img_shape[0])


def ap_per_class(tp, conf, pred_cls, target_cls):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    pr_score = 0.1
    s = [unique_classes.shape[0], tp.shape[1]]
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()
        if n_p == 0 or n_gt == 0:
            continue
        else:
            fpc = (1 - tp[i]).cumsum(axis=0)
            tpc = tp[i].cumsum(axis=0)
            recall = tpc / (n_gt + 1e-16)
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])
            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    mrec = np.concatenate(([0.0], recall, [min(recall[-1] + 0.001, 1.0)]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = "interp"
    if method == "interp":
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    box2 = box2.T
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    inter = (paddle.minimum(b1_x2, b2_x2) - paddle.maximum(b1_x1, b2_x1)).clip(
        min=0
    ) * (paddle.minimum(b1_y2, b2_y2) - paddle.maximum(b1_y1, b2_y1)).clip(min=0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + 1e-16 + w2 * h2 - inter
    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = paddle.maximum(b1_x2, b2_x2) - paddle.minimum(b1_x1, b2_x1)
        ch = paddle.maximum(b1_y2, b2_y2) - paddle.minimum(b1_y1, b2_y1)
        if GIoU:
            c_area = cw * ch + 1e-16
            return iou - (c_area - union) / c_area
        if DIoU or CIoU:
            c2 = cw**2 + ch**2 + 1e-16
            rho2 = (b2_x1 + b2_x2 - (b1_x1 + b1_x2)) ** 2 / 4 + (
                b2_y1 + b2_y2 - (b1_y1 + b1_y2)
            ) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (
                    4
                    / math.pi**2
                    * paddle.pow(x=paddle.atan(x=w2 / h2) - paddle.atan(x=w1 / h1), y=2)
                )
                with paddle.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)
    return iou


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (
        (
            paddle.minimum(box1[:, None, 2:], box2[:, 2:])
            - paddle.maximum(box1[:, None, :2], box2[:, :2])
        )
        .clip(min=0)
        .prod(axis=2)
    )
    return inter / (area1[:, None] + area2 - inter)


def wh_iou(wh1, wh2):
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = paddle.minimum(wh1, wh2).prod(axis=2)
    return inter / (wh1.prod(axis=2) + wh2.prod(axis=2) - inter)


class FocalLoss(paddle.nn.Layer):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = paddle.nn.functional.sigmoid(x=pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(paddle.nn.Layer):
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = paddle.nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = paddle.nn.functional.sigmoid(x=pred)
        dx = pred - true
        alpha_factor = 1 - paddle.exp(x=(dx - 1) / (self.alpha + 0.0001))
        loss *= alpha_factor
        return loss.mean()


def compute_loss(p, targets, model):
    device = targets.place
    lcls, lbox, lobj = (
        paddle.zeros(shape=[1]),
        paddle.zeros(shape=[1]),
        paddle.zeros(shape=[1]),
    )
    tcls, tbox, indices, anchors = build_targets(p, targets, model)
    h = model.hyp
    BCEcls = paddle.nn.BCEWithLogitsLoss(
        pos_weight=paddle.to_tensor(data=[h["cls_pw"]], dtype="float32")
    ).to(device)
    BCEobj = paddle.nn.BCEWithLogitsLoss(
        pos_weight=paddle.to_tensor(data=[h["obj_pw"]], dtype="float32")
    ).to(device)
    cp, cn = smooth_BCE(eps=0.0)
    g = h["fl_gamma"]
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
    nt = 0
    np = len(p)
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if np == 5 else balance
    for i, pi in enumerate(p):
        b, a, gj, gi = indices[i]
        tobj = paddle.zeros_like(x=pi[..., 0])
        n = b.shape[0]
        if n:
            nt += n
            ps = pi[b, a, gj, gi]
            pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = paddle.concat(x=(pxy, pwh), axis=1).to(device)
            giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
            lbox += (1.0 - giou).mean()
            tobj[b, a, gj, gi] = (
                1.0 - model.gr + model.gr * giou.detach().clip(min=0).astype(tobj.dtype)
            )
            if model.nc > 1:
                t = paddle.full_like(x=ps[:, 5:], fill_value=cn)
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)
        lobj += BCEobj(pi[..., 4], tobj) * balance[i]
    s = 3 / np
    lbox *= h["giou"] * s
    lobj *= h["obj"] * s * (1.4 if np >= 4 else 1.0)
    lcls *= h["cls"] * s
    bs = tobj.shape[0]
    loss = lbox + lobj + lcls
    return loss * bs, paddle.concat(x=(lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]
    na, nt = det.na, targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = paddle.ones(shape=[7])
    ai = paddle.arange(end=na).astype(dtype="float32").reshape((na, 1)).repeat(1, nt)
    targets = paddle.concat(x=(targets.repeat(na, 1, 1), ai[:, :, None]), axis=2)
    g = 0.5
    off = (
        paddle.to_tensor(
            data=[[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], place=targets.place
        ).astype(dtype="float32")
        * g
    )
    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = paddle.to_tensor(data=p[i].shape)[[3, 2, 3, 2]]
        t = targets * gain
        if nt:
            r = t[:, :, 4:6] / anchors[:, None]
            j = paddle.maximum(r, 1.0 / r).maximum(2)[0] < model.hyp["anchor_t"]
            t = t[j]
            gxy = t[:, 2:4]
            gxi = gain[[2, 3]] - gxy
            j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
            l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
            j = paddle.stack(x=(paddle.ones_like(x=j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (paddle.zeros_like(x=gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0
        b, c = t[:, :2].astype(dtype="int64").T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        gij = (gxy - offsets).astype(dtype="int64")
        gi, gj = gij.T
        a = t[:, 6].astype(dtype="int64")
        indices.append(
            (b, a, gj.clip_(min=0, max=gain[3]), gi.clip_(min=0, max=gain[2]))
        )
        tbox.append(paddle.concat(x=(gxy - gij, gwh), axis=1))
        anch.append(anchors[a])
        tcls.append(c)
    return tcls, tbox, indices, anch


def non_max_suppression(
    prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False
):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is "float16":
        prediction = prediction.astype(dtype="float32")
    nc = prediction[0].shape[1] - 5
    xc = prediction[..., 4] > conf_thres
    min_wh, max_wh = 2, 4096
    max_det = 300
    time_limit = 10.0
    redundant = True
    multi_label = nc > 1
    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = paddle.concat(
                x=(box[i], x[i, j + 5, None], j[:, None].astype(dtype="float32")),
                axis=1,
            )
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = paddle.concat(x=(box, conf, j.astype(dtype="float32")), axis=1)[
                conf.reshape((-1)) > conf_thres
            ]
        if classes:
            x = x[
                (x[:, 5:6] == paddle.to_tensor(data=classes, place=x.place))
                .astype("bool")
                .any(axis=1)
            ]
        n = x.shape[0]
        if not n:
            continue
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = paddle.vision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and 1 < n < 3000.0:
            try:
                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = paddle.mm(input=weights, mat2=x[:, :4]).astype(
                    dtype="float32"
                ) / weights.sum(axis=1, keepdim=True)
                if redundant:
                    i = i[iou.sum(axis=1) > 1]
            except:
                print(x, i, x.shape, i.shape)
                pass
        output[xi] = x[i]
        if time.time() - t > time_limit:
            break
    return output


def intersect(box_a, box_b):
    n = box_a.shape[0]
    A = box_a.shape[1]
    B = box_b.shape[1]
    max_xy = paddle.minimum(
        box_a[:, :, 2:].unsqueeze(axis=2).expand(shape=[n, A, B, 2]),
        box_b[:, :, 2:].unsqueeze(axis=1).expand(shape=[n, A, B, 2]),
    )
    min_xy = paddle.maximum(
        box_a[:, :, :2].unsqueeze(axis=2).expand(shape=[n, A, B, 2]),
        box_b[:, :, :2].unsqueeze(axis=1).expand(shape=[n, A, B, 2]),
    )
    inter = paddle.clip(x=max_xy - min_xy, min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]))
        .unsqueeze(axis=2)
        .expand_as(y=inter)
    )
    area_b = (
        ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]))
        .unsqueeze(axis=1)
        .expand_as(y=inter)
    )
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / (union + 1e-07)
    return out if use_batch else out.squeeze(axis=0)


def jaccard_diou(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]))
        .unsqueeze(axis=2)
        .expand_as(y=inter)
    )
    area_b = (
        ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]))
        .unsqueeze(axis=1)
        .expand_as(y=inter)
    )
    union = area_a + area_b - inter
    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(axis=2).expand_as(y=inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(axis=2).expand_as(y=inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(axis=1).expand_as(y=inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(axis=1).expand_as(y=inter)
    t1 = box_a[:, :, 1].unsqueeze(axis=2).expand_as(y=inter)
    b1 = box_a[:, :, 3].unsqueeze(axis=2).expand_as(y=inter)
    l1 = box_a[:, :, 0].unsqueeze(axis=2).expand_as(y=inter)
    r1 = box_a[:, :, 2].unsqueeze(axis=2).expand_as(y=inter)
    t2 = box_b[:, :, 1].unsqueeze(axis=1).expand_as(y=inter)
    b2 = box_b[:, :, 3].unsqueeze(axis=1).expand_as(y=inter)
    l2 = box_b[:, :, 0].unsqueeze(axis=1).expand_as(y=inter)
    r2 = box_b[:, :, 2].unsqueeze(axis=1).expand_as(y=inter)
    cr = paddle.maximum(r1, r2)
    cl = paddle.minimum(l1, l2)
    ct = paddle.minimum(t1, t2)
    cb = paddle.maximum(b1, b2)
    D = ((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((cr - cl) ** 2 + (cb - ct) ** 2 + 1e-07)
    out = inter / area_a if iscrowd else inter / (union + 1e-07) - D**0.7
    return out if use_batch else out.squeeze(axis=0)


def box_diou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())
    lt = paddle.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = paddle.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    clt = paddle.minimum(boxes1[:, None, :2], boxes2[:, :2])
    crb = paddle.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    x1 = (boxes1[:, None, 0] + boxes1[:, None, 2]) / 2
    y1 = (boxes1[:, None, 1] + boxes1[:, None, 3]) / 2
    x2 = (boxes2[:, None, 0] + boxes2[:, None, 2]) / 2
    y2 = (boxes2[:, None, 1] + boxes2[:, None, 3]) / 2
    d = (x1 - x2.t()) ** 2 + (y1 - y2.t()) ** 2
    c = ((crb - clt) ** 2).sum(axis=2)
    inter = (rb - lt).clip(min=0).prod(axis=2)
    return inter / (area1[:, None] + area2 - inter) - (d / c) ** 0.6


def non_max_suppression2(
    prediction,
    conf_thres=0.1,
    iou_thres=0.6,
    max_box=1500,
    merge=False,
    classes=None,
    agnostic=False,
):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is "float16":
        prediction = prediction.astype(dtype="float32")
    nc = prediction[0].shape[1] - 5
    xc = prediction[..., 4] > conf_thres
    min_wh, max_wh = 2, 4096
    max_det = 300
    time_limit = 10.0
    redundant = True
    multi_label = nc > 1
    t = time.time()
    output = [None] * prediction.shape[0]
    pred1 = (prediction < -1).astype(dtype="float32")[:, :max_box, :6]
    pred2 = pred1[:, :, :4] + 0
    batch_size = prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = paddle.concat(
                x=(box[i], x[i, j + 5, None], j[:, None].astype(dtype="float32")),
                axis=1,
            )
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = paddle.concat(x=(box, conf, j.astype(dtype="float32")), axis=1)[
                conf.reshape((-1)) > conf_thres
            ]
        if classes:
            x = x[
                (x[:, 5:6] == paddle.to_tensor(data=classes, place=x.place))
                .astype("bool")
                .any(axis=1)
            ]
        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)]
        c = x[:, 5] * 0 if agnostic else x[:, 5]
        boxes = (x[:, :4].clone() + c.reshape((-1, 1)) * max_wh)[:max_box]
        pred2[xi, :] = paddle.concat(x=(boxes, pred2[xi, :]), axis=0)[:max_box]
        pred1[xi, :] = paddle.concat(x=(x[:max_box], pred1[xi, :]), axis=0)[:max_box]
    iou = jaccard_diou(pred2, pred2).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A = B
        maxA = A.max(dim=1)[0]
        E = (maxA < iou_thres).astype(dtype="float32").unsqueeze(axis=2).expand_as(y=A)
        B = iou.mul(E)
        if A.equal_all(y=B).item() == True:
            break
    keep = maxA <= iou_thres
    weights = (
        B * (B > 0.8)
        + paddle.eye(num_rows=max_box).expand(shape=[batch_size, max_box, max_box])
    ) * pred1[:, :, 4].reshape((batch_size, 1, max_box))
    pred1[:, :, :4] = paddle.matmul(x=weights, y=pred1[:, :, :4]) / weights.sum(
        axis=2, keepdim=True
    )
    for jj in range(batch_size):
        output[jj] = pred1[jj][keep[jj]]
    return output


def strip_optimizer(f="weights/best.pt", s=""):
    x = paddle.load(path=f)
    x["optimizer"] = None
    x["training_results"] = None
    x["epoch"] = -1
    x["model"].astype(dtype="float16")
    for p in x["model"].parameters():
        p.stop_gradient = not False
    paddle.save(obj=x, path=s or f)
    mb = os.path.getsize(s or f) / 1000000.0
    print(
        "Optimizer stripped from %s,%s %.1fMB"
        % (f, " saved as %s," % s if s else "", mb)
    )


def coco_class_count(path="../coco/labels/train2014/"):
    nc = 80
    x = np.zeros(nc, dtype="int32")
    files = sorted(glob.glob("%s/*.*" % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype("int32"), minlength=nc)
        print(i, len(files))


def coco_only_people(path="../coco/labels/train2017/"):
    files = sorted(glob.glob("%s/*.*" % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def crop_images_random(path="../images/", scale=0.5):
    for file in tqdm(sorted(glob.glob("%s/*.*" % path))):
        img = cv2.imread(file)
        if img is not None:
            h, w = img.shape[:2]
            a = 30
            mask_h = random.randint(a, int(max(a, h * scale)))
            mask_w = mask_h
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


def coco_single_class_labels(path="../coco/labels/train2014/", label_class=43):
    if os.path.exists("new/"):
        shutil.rmtree("new/")
    os.makedirs("new/")
    os.makedirs("new/labels/")
    os.makedirs("new/images/")
    for file in tqdm(sorted(glob.glob("%s/*.*" % path))):
        with open(file, "r") as f:
            labels = np.array(
                [x.split() for x in f.read().splitlines()], dtype=np.float32
            )
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace("labels", "images").replace("txt", "jpg")
            labels[:, 0] = 0
            with open("new/images.txt", "a") as f:
                f.write(img_file + "\n")
            with open("new/labels/" + Path(file).name, "a") as f:
                for l in labels[i]:
                    f.write("%g %.6f %.6f %.6f %.6f\n" % tuple(l))
            shutil.copyfile(
                src=img_file, dst="new/images/" + Path(file).name.replace("txt", "jpg")
            )


def kmean_anchors(
    path="./data/coco128.yaml", n=9, img_size=640, thr=4.0, gen=1000, verbose=True
):
    """Creates kmeans-evolved anchors from training dataset

    Arguments:
        path: path to dataset *.yaml, or a loaded dataset
        n: number of anchors
        img_size: image size used for training
        thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
        gen: generations to evolve anchors using genetic algorithm

    Return:
        k: kmeans evolved anchors

    Usage:
        from utils.utils import *; _ = kmean_anchors()
    """
    thr = 1.0 / thr

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = paddle.minimum(r, 1.0 / r).min(2)[0]
        return x, x.max(1)[0]

    def fitness(k):
        _, best = metric(paddle.to_tensor(data=k, dtype="float32"), wh)
        return (best * (best > thr).astype(dtype="float32")).mean()

    def print_results(k):
        k = k[np.argsort(k.prod(axis=1))]
        x, best = metric(k, wh0)
        bpr, aat = (
            (best > thr).astype(dtype="float32").mean(),
            (x > thr).astype(dtype="float32").mean() * n,
        )
        print(
            "thr=%.2f: %.4f best possible recall, %.2f anchors past thr"
            % (thr, bpr, aat)
        )
        print(
            "n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: "
            % (n, img_size, x.mean(), best.mean(), x[x > thr].mean()),
            end="",
        )
        for i, x in enumerate(k):
            print(
                "%i,%i" % (round(x[0]), round(x[1])),
                end=",  " if i < len(k) - 1 else "\n",
            )
        return k

    if isinstance(path, str):
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
        from utils.datasets import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(data_dict["train"], augment=True, rect=True)
    else:
        dataset = path
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([(l[:, 3:5] * s) for s, l in zip(shapes, dataset.labels)])
    i = (wh0 < 3.0).astype("bool").any(axis=1).sum()
    if i:
        print(
            "WARNING: Extremely small objects found. %g of %g labels are < 3 pixels in width or height."
            % (i, len(wh0))
        )
    wh = wh0[(wh0 >= 2.0).astype("bool").any(axis=1)]
    print("Running kmeans for %g anchors on %g points..." % (n, len(wh)))
    s = wh.std(axis=0)
    k, dist = kmeans(wh / s, n, iter=30)
    k *= s
    wh = paddle.to_tensor(data=wh, dtype="float32")
    wh0 = paddle.to_tensor(data=wh0, dtype="float32")
    k = print_results(k)
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1
    pbar = tqdm(range(gen), desc="Evolving anchors with Genetic Algorithm")
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).astype("bool").all():
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(
                0.3, 3.0
            )
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = "Evolving anchors with Genetic Algorithm: fitness = %.4f" % f
            if verbose:
                print_results(k)
    return print_results(k)


def print_mutation(hyp, results, yaml_file="hyp_evolved.yaml", bucket=""):
    a = "%10s" * len(hyp) % tuple(hyp.keys())
    b = "%10.3g" * len(hyp) % tuple(hyp.values())
    c = "%10.4g" * len(results) % results
    print("\n%s\n%s\nEvolved fitness: %s\n" % (a, b, c))
    if bucket:
        os.system("gsutil cp gs://%s/evolve.txt ." % bucket)
    with open("evolve.txt", "a") as f:
        f.write(c + b + "\n")
    x = np.unique(np.loadtxt("evolve.txt", ndmin=2), axis=0)
    x = x[np.argsort(-fitness(x))]
    np.savetxt("evolve.txt", x, "%10.3g")
    if bucket:
        os.system("gsutil cp evolve.txt gs://%s" % bucket)
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, "w") as f:
        results = tuple(x[0, :7])
        c = "%10.4g" * len(results) % results
        f.write(
            "# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: "
            % len(x)
            + c
            + "\n\n"
        )
        yaml.dump(hyp, f, sort_keys=False)


def apply_classifier(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
            d = d.clone()
            b = xyxy2xywh(d[:, :4])
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(axis=1)
            b[:, 2:] = b[:, 2:] * 1.3 + 30
            d[:, :4] = xywh2xyxy(b).astype(dtype="int64")
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            pred_cls1 = d[:, 5].astype(dtype="int64")
            ims = []
            for j, a in enumerate(d):
                cutout = im0[i][int(a[1]) : int(a[3]), int(a[0]) : int(a[2])]
                im = cv2.resize(cutout, (224, 224))
                im = im[:, :, ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im, dtype=np.float32)
                im /= 255.0
                ims.append(im)
            pred_cls2 = model(paddle.to_tensor(data=ims).to(d.place)).argmax(axis=1)
            x[i] = x[i][pred_cls1 == pred_cls2]
    return x


def fitness(x):
    w = [0.0, 0.0, 0.1, 0.9]
    return (x[:, :4] * w).sum(axis=1)


def output_to_target(output, width, height):
    if isinstance(output, paddle.Tensor):
        output = output.cpu().numpy()
    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])
                targets.append([i, cls, x, y, w, h, conf])
    return np.array(targets)


def increment_dir(dir, comment=""):
    n = 0
    dir = str(Path(dir))
    d = sorted(glob.glob(dir + "*"))
    if len(d):
        n = max([int(x[len(dir) : x.find("_") if "_" in x else None]) for x in d]) + 1
    return dir + str(n) + ("_" + comment if comment else "")


def hist2d(x, y, n=100):
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_wh_methods():
    x = np.arange(-4.0, 4.0, 0.1)
    ya = np.exp(x)
    yb = paddle.nn.functional.sigmoid(x=paddle.to_tensor(data=x)).numpy() * 2
    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, ".-", label="YOLO")
    plt.plot(x, yb**2, ".-", label="YOLO ^2")
    plt.plot(x, yb**1.6, ".-", label="YOLO ^1.6")
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel("input")
    plt.ylabel("output")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=200)


def plot_images(
    images,
    targets,
    paths=None,
    fname="images.jpg",
    names=None,
    max_size=640,
    max_subplots=16,
):
    tl = 3
    tf = max(tl - 1, 1)
    if os.path.isfile(fname):
        return None
    if isinstance(images, paddle.Tensor):
        images = images.cpu().astype(dtype="float32").numpy()
    if isinstance(targets, paddle.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs**0.5)
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    hex2rgb = lambda h: tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()["color"]]
    for i, img in enumerate(images):
        if i == max_subplots:
            break
        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))
        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))
        mosaic[block_y : block_y + h, block_x : block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype("int")
            gt = image_targets.shape[1] == 6
            conf = None if gt else image_targets[:, 6]
            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:
                    label = "%s" % cls if gt else "%s %.1f" % (cls, conf[j])
                    plot_one_box(
                        box, mosaic, label=label, color=color, line_thickness=tl
                    )
        if paths is not None:
            label = os.path.basename(paths[i])[:40]
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(
                mosaic,
                label,
                (block_x + 5, block_y + t_size[1] + 5),
                0,
                tl / 3,
                [220, 220, 220],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        cv2.rectangle(
            mosaic,
            (block_x, block_y),
            (block_x + w, block_y + h),
            (255, 255, 255),
            thickness=3,
        )
    if fname is not None:
        mosaic = cv2.resize(
            mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=""):
    optimizer, scheduler = copy(optimizer), copy(scheduler)
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]["lr"])
    plt.plot(y, ".-", label="LR")
    plt.xlabel("epoch")
    plt.ylabel("LR")
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "LR.png", dpi=200)


def plot_test_txt():
    x = np.loadtxt("test.txt", dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect("equal")
    plt.savefig("hist2d.png", dpi=300)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig("hist1d.png", dpi=200)


def plot_targets_txt():
    x = np.loadtxt("targets.txt", dtype=np.float32).T
    s = ["x targets", "y targets", "width targets", "height targets"]
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.flatten()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label="%.3g +/- %.3g" % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig("targets.jpg", dpi=200)


def plot_study_txt(f="study.txt", x=None):
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    ax = ax.flatten()
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in [""]:
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = [
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
            "t_inference (ms/img)",
            "t_NMS (ms/img)",
            "t_total (ms/img)",
        ]
        for i in range(7):
            ax[i].plot(x, y[i], ".-", linewidth=2, markersize=8)
            ax[i].set_title(s[i])
        j = y[3].argmax() + 1
        ax2.plot(
            y[6, :j],
            y[3, :j] * 100.0,
            ".-",
            linewidth=2,
            markersize=8,
            label=Path(f).stem.replace("study_coco_", "").replace("yolo", "YOLO"),
        )
    ax2.plot(
        1000.0 / np.array([209, 140, 97, 58, 35, 18]),
        [33.8, 39.6, 43.0, 47.5, 49.4, 50.7],
        "k.-",
        linewidth=2,
        markersize=8,
        alpha=0.25,
        label="EfficientDet",
    )
    ax2.grid()
    ax2.set_xlim(0, 30)
    ax2.set_ylim(28, 50)
    ax2.set_yticks(np.arange(30, 55, 5))
    ax2.set_xlabel("GPU Speed (ms/img)")
    ax2.set_ylabel("COCO AP val")
    ax2.legend(loc="lower right")
    plt.savefig("study_mAP_latency.png", dpi=300)
    plt.savefig(f.replace(".txt", ".png"), dpi=200)


def plot_labels(labels, save_dir=""):
    c, b = labels[:, 0], labels[:, 1:].transpose()
    nc = int(c.max() + 1)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.flatten()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel("classes")
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap="jet")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap="jet")
    ax[2].set_xlabel("width")
    ax[2].set_ylabel("height")
    plt.savefig(Path(save_dir) / "labels.png", dpi=200)
    plt.close()


def plot_evolution(yaml_file="runs/evolve/hyp_evolved.yaml"):
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    x = np.loadtxt("evolve.txt", ndmin=2)
    f = fitness(x)
    plt.figure(figsize=(10, 10), tight_layout=True)
    matplotlib.rc("font", **{"size": 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        mu = y[f.argmax()]
        plt.subplot(5, 5, i + 1)
        plt.scatter(
            y, f, c=hist2d(y, f, 20), cmap="viridis", alpha=0.8, edgecolors="none"
        )
        plt.plot(mu, f.max(), "k+", markersize=15)
        plt.title("%s = %.3g" % (k, mu), fontdict={"size": 9})
        if i % 5 != 0:
            plt.yticks([])
        print("%15s: %.3g" % (k, mu))
    plt.savefig("evolve.png", dpi=200)
    print("\nPlot saved as evolve.png")


def plot_results_overlay(start=0, stop=0):
    s = [
        "train",
        "train",
        "train",
        "Precision",
        "mAP@0.5",
        "val",
        "val",
        "val",
        "Recall",
        "mAP@0.5:0.95",
    ]
    t = ["GIoU", "Objectness", "Classification", "P-R", "mAP-F1"]
    for f in sorted(
        glob.glob("results*.txt") + glob.glob("../../Downloads/results*.txt")
    ):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.flatten()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker=".", label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None
        fig.savefig(f.replace(".txt", ".png"), dpi=200)


def plot_results(start=0, stop=0, bucket="", id=(), labels=(), save_dir=""):
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.flatten()
    s = [
        "GIoU",
        "Objectness",
        "Classification",
        "Precision",
        "Recall",
        "val GIoU",
        "val Objectness",
        "val Classification",
        "mAP@0.5",
        "mAP@0.5:0.95",
    ]
    if bucket:
        os.system("rm -rf storage.googleapis.com")
        files = [
            ("https://storage.googleapis.com/%s/results%g.txt" % (bucket, x))
            for x in id
        ]
    else:
        files = glob.glob(str(Path(save_dir) / "results*.txt")) + glob.glob(
            "../../Downloads/results*.txt"
        )
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(
                f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2
            ).T
            n = results.shape[1]
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan
                label = labels[fi] if len(labels) else Path(f).stem
                ax[i].plot(x, y, marker=".", label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
        except:
            print("Warning: Plotting error for %s, skipping file" % f)
    fig.tight_layout()
    ax[1].legend()
    fig.savefig(Path(save_dir) / "results.png", dpi=200)
