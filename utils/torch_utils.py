import paddle
import math
import os
import time
from copy import deepcopy


def init_seeds(seed=0):
    paddle.seed(seed=seed)


def select_device(device="", batch_size=None):
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert paddle.device.cuda.device_count() >= 1, (
            "CUDA unavailable, invalid device %s requested" % device
        )
    cuda = False if cpu_request else paddle.device.cuda.device_count() >= 1
    if cuda:
        c = 1024**2
        ng = paddle.device.cuda.device_count()
        if ng > 1 and batch_size:
            assert batch_size % ng == 0, (
                "batch-size %g not multiple of GPU count %g" % (batch_size, ng)
            )
        x = [paddle.device.cuda.get_device_properties(device=i) for i in range(ng)]
        s = "Using CUDA "
        for i in range(0, ng):
            if i == 1:
                s = " " * len(s)
            print(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                % (s, i, x[i].name, x[i].total_memory / c)
            )
    else:
        print("Using CPU")
    print("")
    return str("cuda:0" if cuda else "cpu").replace("cuda", "gpu")


def time_synchronized():
    paddle.device.cuda.synchronize() if paddle.device.cuda.device_count() >= 1 else None
    return time.time()


def is_parallel(model):
    return type(model) in (paddle.DataParallel, paddle.DataParallel)


def intersect_dicts(da, db, exclude=()):
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def initialize_weights(model):
    for m in model.sublayers():
        t = type(m)
        if t is paddle.nn.Conv2D:
            pass
        elif t is paddle.nn.BatchNorm2D:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in [paddle.nn.LeakyReLU, paddle.nn.ReLU, paddle.nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=paddle.nn.Conv2D):
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    a, b = 0.0, 0.0
    for p in model.parameters():
        a += p.size
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    pass


def fuse_conv_and_bn(conv, bn):
    with paddle.no_grad():
        fusedconv = paddle.nn.Conv2D(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias_attr=True,
        ).to(conv.weight.device)
        w_conv = conv.weight.clone().reshape((conv.out_channels, -1))
        w_bn = paddle.diag(x=bn.weight.div(paddle.sqrt(x=bn.eps + bn.running_var)))
        fusedconv.weight.copy_(
            paddle.mm(input=w_bn, mat2=w_conv).reshape((fusedconv.weight.size()))
        )
        b_conv = (
            paddle.zeros(shape=conv.weight.size(0)) if conv.bias is None else conv.bias
        )
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            paddle.sqrt(x=bn.running_var + bn.eps)
        )
        fusedconv.bias.copy_(
            paddle.mm(input=w_bn, mat2=b_conv.reshape(-1, 1)).reshape(-1) + b_bn
        )
        return fusedconv


def model_info(model, verbose=False):
    n_p = sum(x.size for x in model.parameters())
    n_g = sum(x.size for x in model.parameters() if not x.stop_gradient)
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    not p.stop_gradient,
                    p.size,
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )
    try:
        from thop import profile

        flops = (
            profile(
                deepcopy(model),
                inputs=(paddle.zeros(shape=[1, 3, 64, 64]),),
                verbose=False,
            )[0]
            / 1000000000.0
            * 2
        )
        fs = ", %.1f GFLOPS" % (flops * 100)
    except:
        fs = ""
    print(
        "Model Summary: %g layers, %g parameters, %g gradients%s"
        % (len(list(model.parameters())), n_p, n_g, fs)
    )


def load_classifier(name="resnet101", n=2):
    model = paddle.vision.models.__dict__[name](pretrained=True)
    input_size = [3, 224, 224]
    input_space = "RGB"
    input_range = [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for x in [input_size, input_space, input_range, mean, std]:
        print(x + " =", eval(x))
    filters = model.fc.weight.shape[1]
    out_2 = paddle.create_parameter(
        shape=paddle.zeros(shape=n).shape,
        dtype=paddle.zeros(shape=n).numpy().dtype,
        default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=n)),
    )
    out_2.stop_gradient = not True
    model.fc.bias = out_2
    out_3 = paddle.create_parameter(
        shape=paddle.zeros(shape=[n, filters]).shape,
        dtype=paddle.zeros(shape=[n, filters]).numpy().dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.zeros(shape=[n, filters])
        ),
    )
    out_3.stop_gradient = not True
    model.fc.weight = out_3
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False):
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = int(h * ratio), int(w * ratio)
        img = paddle.nn.functional.interpolate(
            x=img, size=s, mode="bilinear", align_corners=False
        )
        if not same_shape:
            gs = 128
            h, w = [(math.ceil(x * ratio / gs) * gs) for x in (h, w)]
        return paddle.nn.functional.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            out_4 = p
            out_4.stop_gradient = not False
            out_4

    def update(self, model):
        with paddle.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        copy_attr(self.ema, model, include, exclude)
