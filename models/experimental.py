import paddle
import paddle.nn as nn
import numpy as np
from models.common import Conv, DWConv
from utils.google_utils import attempt_download


class CrossConv(nn.Layer):
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2D(
            in_channels=c1, out_channels=c_, kernel_size=1, stride=1, bias_attr=False
        )
        self.cv3 = nn.Conv2D(
            in_channels=c_, out_channels=c_, kernel_size=1, stride=1, bias_attr=False
        )
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2D(num_features=2 * c_)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.m = nn.Sequential(
            *[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(paddle.concat(x=(y1, y2), axis=1))))


class Sum(nn.Layer):
    def __init__(self, n, weight=False):
        super(Sum, self).__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            out_0 = paddle.create_parameter(
                shape=(-paddle.arange(start=1.0, end=n) / 2).shape,
                dtype=(-paddle.arange(start=1.0, end=n) / 2).numpy().dtype,
                default_initializer=nn.initializer.Assign(
                    -paddle.arange(start=1.0, end=n) / 2
                ),
            )
            out_0.stop_gradient = not True
            self.w = out_0

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = nn.functional.sigmoid(x=self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Layer):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super(GhostConv, self).__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return paddle.concat(x=[y, self.cv2(y)], axis=1)


class GhostBottleneck(nn.Layer):
    def __init__(self, c1, c2, k, s):
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Layer):
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:
            i = paddle.linspace(start=0, stop=groups - 1e-06, num=c2).floor()
            c_ = [(i == g).sum() for g in range(groups)]
        else:
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()
        self.m = nn.LayerList(
            sublayers=[
                nn.Conv2D(
                    in_channels=c1,
                    out_channels=int(c_[g]),
                    kernel_size=k[g],
                    stride=s,
                    padding=k[g] // 2,
                    bias_attr=False,
                )
                for g in range(groups)
            ]
        )
        self.bn = nn.BatchNorm2D(num_features=c2)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return x + self.act(self.bn(paddle.concat(x=[m(x) for m in self.m], axis=1)))


class Ensemble(nn.LayerList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = paddle.stack(x=y).mean(axis=0)
        return y, None


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(paddle.load(path=w)["model"].astype(dtype="float32").fuse().eval())
    if len(model) == 1:
        return model[-1]
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model
