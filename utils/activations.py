import paddle


class Swish(paddle.nn.Layer):
    @staticmethod
    def forward(x):
        return x * paddle.nn.functional.sigmoid(x=x)


class HardSwish(paddle.nn.Layer):
    @staticmethod
    def forward(x):
        return x * paddle.nn.functional.hardtanh(x=x + 3, min=0.0, max=6.0) / 6.0


class MemoryEfficientSwish(paddle.nn.Layer):
    class F(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * paddle.nn.functional.sigmoid(x=x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = paddle.nn.functional.sigmoid(x=x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


class Mish(paddle.nn.Layer):
    @staticmethod
    def forward(x):
        return x * paddle.nn.functional.softplus(x=x).tanh()


class MemoryEfficientMish(paddle.nn.Layer):
    class F(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(
                paddle.nn.functional.tanh(x=paddle.nn.functional.softplus(x=x))
            )

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = paddle.nn.functional.sigmoid(x=x)
            fx = paddle.nn.functional.softplus(x=x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


class FReLU(paddle.nn.Layer):
    def __init__(self, c1, k=3):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=c1,
            out_channels=c1,
            kernel_size=k,
            stride=1,
            padding=1,
            groups=c1,
        )
        self.bn = paddle.nn.BatchNorm2D(num_features=c1)

    def forward(self, x):
        return paddle.maximum(x, self.bn(self.conv(x)))
