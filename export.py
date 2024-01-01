import paddle
from models.yolo import Model

if __name__ == "__main__":
    model = Model(cfg="models/yolov4-p5.yaml")
    x = paddle.randn(shape=[1, 3, 640, 640])
    try:
        x = paddle.static.InputSpec.from_tensor(x)
        paddle.jit.save(model, input_spec=(x,), path="./model")
        print("[JIT] paddle.jit.save successed.")
        exit(0)
    except Exception as e:
        print("[JIT] paddle.jit.save failed.")
        raise e
