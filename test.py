import paddle
import argparse
import glob
import json
import os
import shutil
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class,
    check_file,
    check_img_size,
    compute_loss,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
    clip_coords,
    plot_images,
    xywh2xyxy,
    box_iou,
    output_to_target,
    ap_per_class,
)
from utils.torch_utils import select_device, time_synchronized


def test(
    data,
    weights=None,
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    save_json=False,
    single_cls=False,
    augment=False,
    verbose=False,
    model=None,
    dataloader=None,
    save_dir="",
    merge=False,
    save_txt=False,
):
    training = model is not None
    if training:
        device = next(model.parameters()).place
    else:
        device = select_device(opt.place, batch_size=batch_size)
        merge, save_txt = opt.merge, opt.save_txt
        if save_txt:
            out = Path("inference/output")
            if os.path.exists(out):
                shutil.rmtree(out)
            os.makedirs(out)
        for f in glob.glob(str(Path(save_dir) / "test_batch*.jpg")):
            os.remove(f)
        model = attempt_load(weights, map_location=device)
        imgsz = check_img_size(imgsz, s=model.stride.max())
    half = device.type != "cpu"
    if half:
        model.astype(dtype="float16")
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    nc = 1 if single_cls else int(data["nc"])
    iouv = paddle.linspace(start=0.5, stop=0.95, num=10).to(device)
    niou = iouv.size
    if not training:
        img = paddle.zeros(shape=(1, 3, imgsz, imgsz))
        _ = (
            model(img.astype(dtype="float16") if half else img)
            if device.type != "cpu"
            else None
        )
        path = data["test"] if opt.task == "test" else data["val"]
        dataloader = create_dataloader(
            path,
            imgsz,
            batch_size,
            model.stride.max(),
            opt,
            hyp=None,
            augment=False,
            cache=False,
            pad=0.5,
            rect=True,
        )[0]
    seen = 0
    names = model.names if hasattr(model, "names") else model.module.names
    coco91class = coco80_to_coco91_class()
    s = ("%20s" + "%12s" * 6) % (
        "Class",
        "Images",
        "Targets",
        "P",
        "R",
        "mAP@.5",
        "mAP@.5:.95",
    )
    p, r, f1, mp, mr, map50, map, t0, t1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    loss = paddle.zeros(shape=[3])
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.astype(dtype="float16") if half else img.astype(dtype="float32")
        img /= 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape
        whwh = paddle.to_tensor(
            data=[width, height, width, height], dtype="float32"
        ).to(device)
        with paddle.no_grad():
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)
            t0 += time_synchronized() - t
            if training:
                loss += compute_loss(
                    [x.astype(dtype="float32") for x in train_out], targets, model
                )[1][:3]
            t = time_synchronized()
            output = non_max_suppression(
                inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge
            )
            t1 += time_synchronized() - t
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1
            if pred is None:
                if nl:
                    stats.append(
                        (
                            paddle.zeros(shape=[0, niou], dtype="bool"),
                            paddle.to_tensor(data=[]),
                            paddle.to_tensor(data=[]),
                            tcls,
                        )
                    )
                continue
            if save_txt:
                gn = paddle.to_tensor(data=shapes[si][0])[[1, 0, 1, 0]]
                txt_path = str(out / Path(paths[si]).stem)
                pred[:, :4] = scale_coords(
                    img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1]
                )
                for *xyxy, conf, cls in pred:
                    xywh = (
                        (xyxy2xywh(paddle.to_tensor(data=xyxy).reshape((1, 4))) / gn)
                        .reshape((-1))
                        .tolist()
                    )
                    with open(txt_path + ".txt", "a") as f:
                        f.write(("%g " * 5 + "\n") % (cls, *xywh))
            clip_coords(pred, (height, width))
            if save_json:
                image_id = Path(paths[si]).stem
                box = pred[:, :4].clone()
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])
                box = xyxy2xywh(box)
                box[:, :2] -= box[:, 2:] / 2
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append(
                        {
                            "image_id": int(image_id)
                            if image_id.isnumeric()
                            else image_id,
                            "category_id": coco91class[int(p[5])],
                            "bbox": [round(x, 3) for x in b],
                            "score": round(p[4], 5),
                        }
                    )
            correct = paddle.zeros(shape=[pred.shape[0], niou], dtype="bool")
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                for cls in paddle.unique(x=tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).reshape((-1))
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).reshape((-1))
                    if pi.shape[0]:
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        if batch_i < 1:
            f = Path(save_dir) / ("test_batch%g_gt.jpg" % batch_i)
            plot_images(img, targets, paths, str(f), names)
            f = Path(save_dir) / ("test_batch%g_pred.jpg" % batch_i)
            plot_images(
                img, output_to_target(output, width, height), paths, str(f), names
            )
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].astype("bool").any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(axis=1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = paddle.zeros(shape=[1])
    pf = "%20s" + "%12.3g" * 6
    print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    t = tuple(x / seen * 1000.0 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)
    if not training:
        print(
            "Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g"
            % t
        )
    if save_json and len(jdict):
        f = "detections_val2017_%s_results.json" % (
            weights.split(os.sep)[-1].replace(".pt", "")
            if isinstance(weights, str)
            else ""
        )
        print("\nCOCO mAP with pycocotools... saving %s..." % f)
        with open(f, "w") as file:
            json.dump(jdict, file)
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(glob.glob("../coco/annotations/instances_val*.json")[0])
            cocoDt = cocoGt.loadRes(f)
            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]
        except Exception as e:
            print("ERROR: pycocotools unable to run: %s" % e)
    model.astype(dtype="float32")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="yolov4-p5.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--data", type=str, default="data/coco128.yaml", help="*.data path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.65, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="save a cocoapi-compatible JSON results file",
    )
    parser.add_argument("--task", default="val", help="'val', 'test', 'study'")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--merge", action="store_true", help="use Merge NMS")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.data = check_file(opt.data)
    print(opt)
    if opt.task in ["val", "test"]:
        test(
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
            opt.verbose,
        )
    elif opt.task == "study":
        for weights in [""]:
            f = "study_%s_%s.txt" % (Path(opt.data).stem, Path(weights).stem)
            x = list(range(352, 832, 64))
            y = []
            for i in x:
                print("\nRunning %s point %s..." % (f, i))
                r, _, t = test(
                    opt.data,
                    weights,
                    opt.batch_size,
                    i,
                    opt.conf_thres,
                    opt.iou_thres,
                    opt.save_json,
                )
                y.append(r + t)
            np.savetxt(f, y, fmt="%10.4g")
        os.system("zip -r study.zip study_*.txt")
