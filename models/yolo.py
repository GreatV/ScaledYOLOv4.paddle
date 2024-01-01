import paddle
import paddle.nn as nn
import argparse
import math
from copy import deepcopy
from pathlib import Path
from models.common import (
    Conv,
    DWConv,
    Focus,
    Bottleneck,
    SPP,
    BottleneckCSP,
    BottleneckCSP2,
    SPPCSP,
    VoVCSP,
    Concat,
    HarDBlock,
    HarDBlock2,
)
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file
from utils.torch_utils import (
    time_synchronized,
    fuse_conv_and_bn,
    model_info,
    scale_img,
    initialize_weights,
    select_device,
)


class Detect(nn.Layer):
    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.stride = None
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [paddle.zeros(shape=[1])] * self.nl
        a = paddle.to_tensor(data=anchors).astype(dtype="float32").reshape((self.nl, -1, 2))
        self.register_buffer(name="anchors", tensor=a)
        self.register_buffer(
            name="anchor_grid", tensor=a.clone().reshape((self.nl, 1, -1, 1, 1, 2))
        )
        self.m = nn.LayerList(
            sublayers=(
                nn.Conv2D(
                    in_channels=x, out_channels=self.no * self.na, kernel_size=1
                )
                for x in ch
            )
        )
        self.export = False

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i].reshape((bs, self.na, self.no, ny, nx)).transpose(perm=[0, 1, 3, 4, 2])
            )
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].place)
                y = x[i].sigmoid()
                y[..., 0:2] = (
                    y[..., 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].place)
                ) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.reshape((bs, -1, self.no)))
        return x if self.training else (paddle.concat(x=z, axis=1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = paddle.meshgrid([paddle.arange(end=ny), paddle.arange(end=nx)])
        return (
            paddle.stack(x=(xv, yv), axis=2)
            .reshape(((1, 1, ny, nx, 2)))
            .astype(dtype="float32")
        )


class Model(nn.Layer):
    def __init__(self, cfg="yolov4-p5.yaml", ch=3, nc=None):
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)
        if nc and nc != self.yaml["nc"]:
            print("Overriding %s nc=%g with nc=%g" % (cfg, self.yaml["nc"], nc))
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = paddle.to_tensor(
                data=[
                    (s / x.shape[-2])
                    for x in self.forward(paddle.zeros(shape=[1, ch, s, s]))
                ]
            )
            m.anchors /= m.stride.reshape((-1, 1, 1))
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)
        self.info()
        print("")

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]
            s = [1, 0.83, 0.67]
            f = [None, 3, None]
            y = []
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(axis=fi) if fi else x, si)
                yi = self.forward_once(xi)[0]
                yi[..., :4] /= si
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]
                y.append(yi)
            return paddle.concat(x=y, axis=1), None
        else:
            return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [(x if j == -1 else y[j]) for j in m.f]
                )
            if profile:
                try:
                    import thop

                    o = (
                        thop.profile(m, inputs=(x,), verbose=False)[0]
                        / 1000000000.0
                        * 2
                    )
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print("%10.1f%10.0f%10.1fms %-40s" % (o, m.np, dt[-1], m.type))
            x = m(x)
            y.append(x if m.i in self.save else None)
        if profile:
            print("%.1fms total" % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.reshape((m.na, -1))
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else paddle.log(x=cf / cf.sum())
            )
            out_1 = paddle.create_parameter(
                shape=b.reshape((-1,)).shape,
                dtype=b.reshape((-1,)).numpy().dtype,
                default_initializer=nn.initializer.Assign(b.reshape((-1,))),
            )
            out_1.stop_gradient = not True
            mi.bias = out_1

    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.detach().reshape((m.na, -1)).T
            print(
                ("%6g Conv2d.bias:" + "%10.3g" * 6)
                % (mi.weight.shape[1], *b[:5].mean(axis=1).tolist(), b[5:].mean())
            )

    def fuse(self):
        print("Fusing layers... ", end="")
        for m in self.model.sublayers():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                m.bn = None
                m.forward = m.fuseforward
        self.info()
        return self

    def info(self):
        model_info(self)


def parse_model(d, ch):
    print(
        "\n%3s%18s%3s%10s  %-40s%-30s"
        % ("", "from", "n", "params", "module", "arguments")
    )
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [
            nn.Conv2D,
            Conv,
            Bottleneck,
            SPP,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            BottleneckCSP2,
            SPPCSP,
            VoVCSP,
            C3,
        ]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m in [HarDBlock, HarDBlock2]:
            c1 = ch[f]
            args = [c1, *args[:]]
        elif m is nn.BatchNorm2D:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum([x.size for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        print("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if m in [HarDBlock, HarDBlock2]:
            c2 = m_.get_out_ch()
            ch.append(c2)
        else:
            ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov4-p5.yaml", help="model.yaml")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    device = select_device(opt.place)
    model = Model(opt.cfg).to(device)
    model.train()
