import paddle
import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
from PIL import Image, ExifTags
from tqdm import tqdm
from utils.general import xyxy2xywh, xywh2xyxy, torch_distributed_zero_first

help_url = ""
img_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
vid_formats = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(files):
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = s[1], s[0]
        elif rotation == 8:
            s = s[1], s[0]
    except:
        pass
    return s


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    opt,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    local_rank=-1,
    world_size=1,
):
    with torch_distributed_zero_first(local_rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache_images=cache,
            single_cls=opt.single_cls,
            stride=int(stride),
            pad=pad,
        )
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])
    train_sampler = (
        paddle.io.DistributedBatchSampler(dataset=dataset, shuffle=True, batch_size=1)
        if local_rank != -1
        else None
    )
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    return dataloader, dataset


class LoadImages:
    def __init__(self, path, img_size=640):
        p = str(Path(path))
        p = os.path.abspath(p)
        if "*" in p:
            files = sorted(glob.glob(p))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception("ERROR: %s does not exist" % p)
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)
        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "images"
        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None
        assert self.nf > 0, """No images or videos found in %s. Supported formats are:
images: %s
videos: %s""" % (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            self.mode = "video"
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            self.frame += 1
            print(
                "video %g/%g (%g/%g) %s: "
                % (self.count + 1, self.nf, self.frame, self.nframes, path),
                end="",
            )
        else:
            self.count += 1
            img0 = cv2.imread(path)
            assert img0 is not None, "Image Not Found " + path
            print("image %g/%g %s: " % (self.count, self.nf, path), end="")
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf


class LoadWebcam:
    def __init__(self, pipe=0, img_size=640):
        self.img_size = img_size
        if pipe == "0":
            pipe = 0
        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord("q"):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration
        if self.pipe == 0:
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)
        else:
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break
        assert ret_val, "Camera Error %s" % self.pipe
        img_path = "webcam.jpg"
        print("webcam %g: " % self.count, end="")
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:
    def __init__(self, sources="streams.txt", img_size=640):
        self.mode = "images"
        self.img_size = img_size
        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]
        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            print("%g/%g: %s... " % (i + 1, n, s), end="")
            cap = cv2.VideoCapture(0 if s == "0" else s)
            assert cap.isOpened(), "Failed to open %s" % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()
            thread = Thread(target=self.update, args=[i, cap], daemon=True)
            print(" success (%gx%g at %.2f FPS)." % (w, h, fps))
            thread.start()
        print("")
        s = np.stack(
            [letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0
        )
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            print(
                "WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams."
            )

    def update(self, index, cap):
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            if n == 4:
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)
        return self.sources, img, img0, None

    def __len__(self):
        return 0


class LoadImagesAndLabels(paddle.io.Dataset):
    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
    ):
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))
                parent = str(Path(p).parent) + os.sep
                if os.path.isfile(p):
                    with open(p, "r") as t:
                        t = t.read().splitlines()
                        f += [
                            (x.replace("./", parent) if x.startswith("./") else x)
                            for x in t
                        ]
                elif os.path.isdir(p):
                    f += glob.iglob(p + os.sep + "*.*")
                else:
                    raise Exception("%s does not exist" % p)
            self.img_files = sorted(
                [
                    x.replace("/", os.sep)
                    for x in f
                    if os.path.splitext(x)[-1].lower() in img_formats
                ]
            )
        except Exception as e:
            raise Exception(
                "Error loading data from %s: %s\nSee %s" % (path, e, help_url)
            )
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        nb = bi[-1] + 1
        self.n = n
        self.batch = bi
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.label_files = [
            x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
            for x in self.img_files
        ]
        cache_path = str(Path(self.label_files[0]).parent) + ".cache"
        if os.path.isfile(cache_path):
            cache = paddle.load(path=cache_path)
            if cache["hash"] != get_hash(self.label_files + self.img_files):
                cache = self.cache_labels(cache_path)
        else:
            cache = self.cache_labels(cache_path)
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)
        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int)
                * stride
            )
        create_datasubset, extract_bounding_boxes, labels_loaded = (False, False, False)
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0
        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            l = self.labels[i]
            if l.shape[0]:
                assert l.shape[1] == 5, "> 5 label columns: %s" % file
                assert (l >= 0).astype("bool").all(), "negative labels: %s" % file
                assert (l[:, 1:] <= 1).astype("bool").all(), (
                    "non-normalized or out of bounds coordinate labels: %s" % file
                )
                if np.unique(l, axis=0).shape[0] < l.shape[0]:
                    nd += 1
                if single_cls:
                    l[:, 0] = 0
                self.labels[i] = l
                nf += 1
                if create_datasubset and ns < 10000.0:
                    if ns == 0:
                        create_folder(path="./datasubset")
                        os.makedirs("./datasubset/images")
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        with open("./datasubset/images.txt", "a") as f:
                            f.write(self.img_files[i] + "\n")
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = "%s%sclassifier%s%g_%g_%s" % (
                            p.parent.parent,
                            os.sep,
                            os.sep,
                            x[0],
                            j,
                            p.name,
                        )
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)
                        b = x[1:] * [w, h, w, h]
                        b[2:] = b[2:].max()
                        b[2:] = b[2:] * 1.3 + 30
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(
                            f, img[b[1] : b[3], b[0] : b[2]]
                        ), "Failure extracting classifier boxes"
            else:
                ne += 1
            pbar.desc = (
                "Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)"
                % (cache_path, nf, nm, ne, nd, n)
            )
        if nf == 0:
            s = "WARNING: No labels found in %s. See %s" % (
                os.path.dirname(file) + os.sep,
                help_url,
            )
            print(s)
            assert not augment, "%s. Can not train without labels." % s
        self.imgs = [None] * n
        if cache_images:
            gb = 0
            pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = "Caching images (%.1fGB)" % (gb / 1000000000.0)

    def cache_labels(self, path="labels.cache"):
        x = {}
        pbar = tqdm(
            zip(self.img_files, self.label_files),
            desc="Scanning images",
            total=len(self.img_files),
        )
        for img, label in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()
                shape = exif_size(image)
                assert (shape[0] > 9) & (shape[1] > 9), "image size <10 pixels"
                if os.path.isfile(label):
                    with open(label, "r") as f:
                        l = np.array(
                            [x.split() for x in f.read().splitlines()], dtype=np.float32
                        )
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = None
                print("WARNING: %s: %s" % (img, e))
        x["hash"] = get_hash(self.label_files + self.img_files)
        paddle.save(obj=x, path=path)
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]
        hyp = self.hyp
        if self.mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None
            if random.random() < hyp["mixup"]:
                img2, labels2 = load_mosaic(
                    self, random.randint(0, len(self.labels) - 1)
                )
                r = np.random.beta(8.0, 8.0)
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)
            labels = []
            x = self.labels[index]
            if x.size > 0:
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        if self.augment:
            if not self.mosaic:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])
        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]
        if self.augment:
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
        labels_out = paddle.zeros(shape=(nL, 6))
        if nL:
            labels_out[:, 1:] = paddle.to_tensor(data=labels)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return paddle.to_tensor(data=img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return paddle.stack(x=img, axis=0), paddle.concat(x=label, axis=0), path, shapes


def load_image(self, index):
    img = self.imgs[index]
    if img is None:
        path = self.img_files[index]
        img = cv2.imread(path)
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = (x * r[0] % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def load_mosaic(self, index):
    labels4 = []
    s = self.img_size
    yc, xc = s, s
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)
        if i == 0:
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
    img4, labels4 = random_perspective(
        img4,
        labels4,
        degrees=self.hyp["degrees"],
        translate=self.hyp["translate"],
        scale=self.hyp["scale"],
        shear=self.hyp["shear"],
        perspective=self.hyp["perspective"],
        border=self.mosaic_border,
    )
    return img4, labels4


def replicate(img, labels):
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = (x2 - x1 + (y2 - y1)) / 2
    for i in s.argsort()[: round(s.size * 0.5)]:
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)
    return img, labels


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = new_shape, new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 128), np.mod(dh, 128)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, ratio, (dw, dh)


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    M = T @ S @ R @ P @ C
    if border[0] != 0 or border[1] != 0 or (M != np.eye(3)).astype("bool").any():
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        else:
            xy = xy[:, :2].reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]
    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )


def cutout(image, labels):
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        box2 = box2.transpose()
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
            np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
        ).clip(0)
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16
        return inter_area / box2_area

    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])
            labels = labels[ioa < 0.6]
    return labels


def reduce_img_size(path="path/images", img_size=1024):
    path_new = path + "_reduced"
    create_folder(path_new)
    for f in tqdm(glob.glob("%s/*.*" % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)
            if r < 1.0:
                img = cv2.resize(
                    img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA
                )
            fnew = f.replace(path, path_new)
            cv2.imwrite(fnew, img)
        except:
            print("WARNING: image failure %s" % f)


def recursive_dataset2bmp(dataset="path/dataset_bmp"):
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + "/" + file
            s = Path(file).suffix
            if s == ".txt":
                with open(p, "r") as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, ".bmp")
                with open(p, "w") as f:
                    f.write(lines)
            elif s in formats:
                cv2.imwrite(p.replace(s, ".bmp"), cv2.imread(p))
                if s != ".bmp":
                    os.system("rm '%s'" % p)


def imagelist2folder(path="path/images.txt"):
    create_folder(path[:-4])
    with open(path, "r") as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path="./new"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
