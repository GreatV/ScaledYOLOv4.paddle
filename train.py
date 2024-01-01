import paddle
import argparse
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
import test
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    check_img_size,
    torch_distributed_zero_first,
    labels_to_class_weights,
    plot_labels,
    check_anchors,
    labels_to_image_weights,
    compute_loss,
    plot_images,
    fitness,
    strip_optimizer,
    plot_results,
    get_latest_run,
    check_git_status,
    check_file,
    increment_dir,
    print_mutation,
    plot_evolution,
)
from utils.google_utils import attempt_download
from utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts


def train(hyp, opt, device, tb_writer=None):
    print(f"Hyperparameters {hyp}")
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / "evolve"
    wdir = str(log_dir / "weights") + os.sep
    os.makedirs(wdir, exist_ok=True)
    last = wdir + "last.pt"
    best = wdir + "best.pt"
    results_file = str(log_dir / "results.txt")
    epochs, batch_size, total_batch_size, weights, rank = (
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.weights,
        opt.global_rank,
    )
    with open(log_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    cuda = device.type != "cpu"
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (
        (1, ["item"]) if opt.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        opt.data,
    )
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)
        ckpt = paddle.load(path=weights)
        model = Model(opt.cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)
        exclude = ["anchor"] if opt.cfg else []
        state_dict = ckpt["model"].astype(dtype="float32").state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.set_state_dict(state_dict=state_dict, use_structured_name=False)
        print(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(model.state_dict()), weights)
        )
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)
    nbs = 64
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_parameters():
        v.stop_gradient = not True
        if ".bias" in k:
            pg2.append(v)
        elif ".weight" in k and ".bn" not in k:
            pg1.append(v)
        else:
            pg0.append(v)
    if opt.adam:
        optimizer = paddle.optimizer.Adam(
            parameters=pg0,
            learning_rate=hyp["lr0"],
            beta1=(hyp["momentum"], 0.999)[0],
            beta2=(hyp["momentum"], 0.999)[1],
            weight_decay=0.0,
        )
    else:
        optimizer = paddle.optimizer.Momentum(
            learning_rate=hyp["lr0"],
            parameters=pg0,
            momentum=hyp["momentum"],
            use_nesterov=True,
        )
    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})
    optimizer.add_param_group({"params": pg2})
    print(
        "Optimizer groups: %g .bias, %g conv.weight, %g other"
        % (len(pg2), len(pg1), len(pg0))
    )
    del pg0, pg1, pg2
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0 * 0.8 + 0.2
    tmp_lr = paddle.optimizer.lr.LambdaDecay(
        lr_lambda=lf, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt["optimizer"] is not None:
            optimizer.set_state_dict(state_dict=ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])
        start_epoch = ckpt["epoch"] + 1
        if epochs < start_epoch:
            print(
                "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                % (weights, ckpt["epoch"], epochs)
            )
            epochs += ckpt["epoch"]
        del ckpt, state_dict
    gs = int(max(model.stride))
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
    if cuda and rank == -1 and paddle.device.cuda.device_count() > 1:
        model = paddle.DataParallel(layers=model)
    if opt.sync_bn and cuda and rank != -1:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(layer=model).to(device)
        print("Using SyncBatchNorm()")
    ema = ModelEMA(model) if rank in [-1, 0] else None
    if cuda and rank != -1:
        model = paddle.DataParallel(model)
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        opt,
        hyp=hyp,
        augment=True,
        cache=opt.cache_images,
        rect=opt.rect,
        local_rank=rank,
        world_size=opt.world_size,
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
    nb = len(dataloader)
    assert mlc < nc, (
        "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g"
        % (mlc, nc, opt.data, nc - 1)
    )
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate
        testloader = create_dataloader(
            test_path,
            imgsz_test,
            batch_size,
            gs,
            opt,
            hyp=hyp,
            augment=False,
            cache=opt.cache_images,
            rect=True,
            local_rank=-1,
            world_size=opt.world_size,
        )[0]
    hyp["cls"] *= nc / 80.0
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)
    model.names = names
    if rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = paddle.to_tensor(data=labels[:, 0])
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            tb_writer.add_histogram("classes", c, 0)
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)
    t0 = time.time()
    nw = max(3 * nb, 1000.0)
    maps = np.zeros(nc)
    results = 0, 0, 0, 0, 0, 0, 0
    scheduler.last_epoch = start_epoch - 1
    scaler = paddle.amp.GradScaler(
        enable=cuda, incr_every_n_steps=2000, init_loss_scaling=65536.0
    )
    if rank in [0, -1]:
        print("Image sizes %g train, %g test" % (imgsz, imgsz_test))
        print("Using %g dataloader workers" % dataloader.num_workers)
        print("Starting training for %g epochs..." % epochs)
    for epoch in range(start_epoch, epochs):
        model.train()
        if dataset.image_weights:
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2
                image_weights = labels_to_image_weights(
                    dataset.labels, nc=nc, class_weights=w
                )
                dataset.indices = random.choices(
                    range(dataset.n), weights=image_weights, k=dataset.n
                )
            if rank != -1:
                indices = paddle.zeros(shape=[dataset.n], dtype="int32")
                if rank == 0:
                    indices[:] = paddle.to_tensor(dataset.indices, dtype="int32")
                paddle.distributed.broadcast(tensor=indices, src=0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()
        mloss = paddle.zeros(shape=[4])
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(
                ("\n" + "%10s" * 8)
                % (
                    "Epoch",
                    "gpu_mem",
                    "GIoU",
                    "obj",
                    "cls",
                    "total",
                    "targets",
                    "img_size",
                )
            )
            pbar = tqdm(pbar, total=nb)
        optimizer.clear_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).astype(dtype="float32") / 255.0
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(
                    1, np.interp(ni, xi, [1, nbs / total_batch_size]).round()
                )
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(
                        ni, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lf(epoch)]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [0.9, hyp["momentum"]])
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [(math.ceil(x * sf / gs) * gs) for x in imgs.shape[2:]]
                    imgs = paddle.nn.functional.interpolate(
                        x=imgs, size=ns, mode="bilinear", align_corners=False
                    )
            with paddle.amp.auto_cast(enable=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device), model)
                if rank != -1:
                    loss *= opt.world_size
            scaler.scale(loss).backward()
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.clear_grad()
                if ema is not None:
                    ema.update(model)
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = "%.3gG" % (
                    paddle.device.cuda.memory_reserved() / 1000000000.0
                    if paddle.device.cuda.device_count() >= 1
                    else 0
                )
                s = ("%10s" * 2 + "%10.4g" * 6) % (
                    "%g/%g" % (epoch, epochs - 1),
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                pbar.set_description(s)
                if ni < 3:
                    f = str(log_dir / ("train_batch%g.jpg" % ni))
                    result = plot_images(
                        images=imgs, targets=targets, paths=paths, fname=f
                    )
                    if tb_writer and result is not None:
                        tb_writer.add_image(
                            f, result, dataformats="HWC", global_step=epoch
                        )
        scheduler.step()
        if rank in [-1, 0]:
            if ema is not None:
                ema.update_attr(
                    model, include=["yaml", "nc", "hyp", "gr", "names", "stride"]
                )
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:
                results, maps, times = test.test(
                    opt.data,
                    batch_size=batch_size,
                    imgsz=imgsz_test,
                    save_json=final_epoch and opt.data.endswith(os.sep + "coco.yaml"),
                    model=ema.ema.module if hasattr(ema.ema, "module") else ema.ema,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    save_dir=log_dir,
                )
            with open(results_file, "a") as f:
                f.write(s + "%10.4g" * 7 % results + "\n")
            if len(opt.name) and opt.bucket:
                os.system(
                    "gsutil cp %s gs://%s/results/results%s.txt"
                    % (results_file, opt.bucket, opt.name)
                )
            if tb_writer:
                tags = [
                    "train/giou_loss",
                    "train/obj_loss",
                    "train/cls_loss",
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/giou_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                ]
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            save = not opt.nosave or final_epoch and not opt.evolve
            if save:
                with open(results_file, "r") as f:
                    ckpt = {
                        "epoch": epoch,
                        "best_fitness": best_fitness,
                        "training_results": f.read(),
                        "model": ema.ema.module if hasattr(ema, "module") else ema.ema,
                        "optimizer": None if final_epoch else optimizer.state_dict(),
                    }
                paddle.save(obj=ckpt, path=last)
                if epoch >= epochs - 30:
                    paddle.save(
                        obj=ckpt, path=last.replace(".pt", "_{:03d}.pt".format(epoch))
                    )
                if best_fitness == fi:
                    paddle.save(obj=ckpt, path=best)
                del ckpt
    if rank in [-1, 0]:
        n = ("_" if len(opt.name) and not opt.name.isnumeric() else "") + opt.name
        fresults, flast, fbest = (
            "results%s.txt" % n,
            wdir + "last%s.pt" % n,
            wdir + "best%s.pt" % n,
        )
        for f1, f2 in zip(
            [wdir + "last.pt", wdir + "best.pt", "results.txt"],
            [flast, fbest, fresults],
        ):
            if os.path.exists(f1):
                os.rename(f1, f2)
                ispt = f2.endswith(".pt")
                strip_optimizer(f2, f2.replace(".pt", "_strip.pt")) if ispt else None
                os.system(
                    "gsutil cp %s gs://%s/weights" % (f2, opt.bucket)
                ) if opt.bucket and ispt else None
        if not opt.evolve:
            plot_results(save_dir=log_dir)
        print(
            "%g epochs completed in %.3f hours.\n"
            % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
        )
    paddle.distributed.destroy_process_group() if rank not in [-1, 0] else None
    paddle.device.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov4-p5.pt", help="initial weights path"
    )
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--data", type=str, default="data/coco128.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="",
        help="hyperparameters path, i.e. data/hyp.scratch.yaml",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=[640, 640], help="train,test sizes"
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="get_last",
        default=False,
        help="resume from given path/last.pt, or most recent run if blank",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--name", default="", help="renames results.txt to results_name.txt if supplied"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="train as single-class dataset"
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument("--logdir", type=str, default="runs/", help="logging directory")
    opt = parser.parse_args()
    if opt.resume:
        last = get_latest_run() if opt.resume == "get_last" else opt.resume
        if last and not opt.weights:
            print(f"Resuming training from {last}")
        opt.weights = last if opt.resume and not opt.weights else opt.weights
    if opt.local_rank == -1 or "RANK" in os.environ and os.environ["RANK"] == "0":
        check_git_status()
    opt.hyp = opt.hyp or (
        "data/hyp.finetune.yaml" if opt.weights else "data/hyp.scratch.yaml"
    )
    opt.data, opt.cfg, opt.hyp = (
        check_file(opt.data),
        check_file(opt.cfg),
        check_file(opt.hyp),
    )
    assert len(opt.cfg) or len(
        opt.weights
    ), "either --cfg or --weights must be specified"
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
    device = select_device(opt.place, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    opt.global_rank = -1
    if opt.local_rank != -1:
        assert paddle.device.cuda.device_count() > opt.local_rank
        paddle.device.set_device(device=opt.local_rank)
        device = ":".join(["cuda".replace("cuda", "gpu"), str(opt.local_rank)])
        paddle.distributed.init_parallel_env()
        opt.world_size = paddle.distributed.get_world_size()
        opt.global_rank = paddle.distributed.get_rank()
        assert (
            opt.batch_size % opt.world_size == 0
        ), "--batch-size must be multiple of CUDA device count"
        opt.batch_size = opt.total_batch_size // opt.world_size
    print(opt)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            print(
                'Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/'
                % opt.logdir
            )
        train(hyp, opt, device, tb_writer)
    else:
        meta = {
            "lr0": (1, 1e-05, 0.1),
            "momentum": (0.1, 0.6, 0.98),
            "weight_decay": (1, 0.0, 0.001),
            "giou": (1, 0.02, 0.2),
            "cls": (1, 0.2, 4.0),
            "cls_pw": (1, 0.5, 2.0),
            "obj": (1, 0.2, 4.0),
            "obj_pw": (1, 0.5, 2.0),
            "iou_t": (0, 0.1, 0.7),
            "anchor_t": (1, 2.0, 8.0),
            "fl_gamma": (0, 0.0, 2.0),
            "hsv_h": (1, 0.0, 0.1),
            "hsv_s": (1, 0.0, 0.9),
            "hsv_v": (1, 0.0, 0.9),
            "degrees": (1, 0.0, 45.0),
            "translate": (1, 0.0, 0.9),
            "scale": (1, 0.0, 0.9),
            "shear": (1, 0.0, 10.0),
            "perspective": (1, 0.0, 0.001),
            "flipud": (0, 0.0, 1.0),
            "fliplr": (1, 0.0, 1.0),
            "mixup": (1, 0.0, 1.0),
        }
        assert opt.local_rank == -1, "DDP mode not implemented for --evolve"
        opt.notest, opt.nosave = True, True
        yaml_file = Path("runs/evolve/hyp_evolved.yaml")
        if opt.bucket:
            os.system("gsutil cp gs://%s/evolve.txt ." % opt.bucket)
        for _ in range(100):
            if os.path.exists("evolve.txt"):
                parent = "single"
                x = np.loadtxt("evolve.txt", ndmin=2)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min()
                if parent == "single" or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(axis=0) / w.sum()
                mp, s = 0.9, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):
                    v = (
                        g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1
                    ).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])
                hyp[k] = min(hyp[k], v[2])
                hyp[k] = round(hyp[k], 5)
            results = train(hyp.copy(), opt, device)
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)
        plot_evolution(yaml_file)
        print(
            """Hyperparameter evolution complete. Best results saved as: %s
Command to train a new model with these hyperparameters: $ python train.py --hyp %s"""
            % (yaml_file, yaml_file)
        )
