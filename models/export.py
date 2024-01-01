import paddle
import argparse
from utils.google_utils import attempt_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="./yolov4-p5.pt", help="weights path"
    )
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=[640, 640], help="image size"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    print(opt)
    img = paddle.zeros(shape=(opt.batch_size, 3, *opt.img_size))
    attempt_download(opt.weights)
    model = paddle.load(path=opt.weights)["model"].astype(dtype="float32")
    model.eval()
    model.model[-1].export = True
    y = model(img)
    try:
        print("\nStarting TorchScript export with torch %s..." % paddle.__version__)
        f = opt.weights.replace(".pt", ".torchscript.pt")
        # ts = torch.jit.trace(model, img)
        # ts.save(f)
        print("TorchScript export success, saved as %s" % f)
    except Exception as e:
        print("TorchScript export failure: %s" % e)
    try:
        import onnx

        print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
        f = opt.weights.replace(".pt", ".onnx")
        model.fuse()
        # torch.onnx.export(
        #     model,
        #     img,
        #     f,
        #     verbose=False,
        #     opset_version=12,
        #     input_names=["images"],
        #     output_names=["classes", "boxes"] if y is None else ["output"],
        # )
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))
        print("ONNX export success, saved as %s" % f)
    except Exception as e:
        print("ONNX export failure: %s" % e)
    try:
        import coremltools as ct

        print("\nStarting CoreML export with coremltools %s..." % ct.__version__)
        # model = ct.convert(
        #     ts,
        #     inputs=[
        #         ct.ImageType(
        #             name="images", shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0]
        #         )
        #     ],
        # )
        f = opt.weights.replace(".pt", ".mlmodel")
        model.save(f)
        print("CoreML export success, saved as %s" % f)
    except Exception as e:
        print("CoreML export failure: %s" % e)
    print("\nExport complete. Visualize with https://github.com/lutzroeder/netron.")
