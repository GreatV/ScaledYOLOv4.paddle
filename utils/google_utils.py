import os
import platform
import time
from pathlib import Path


def attempt_download(weights):
    weights = weights.strip().replace("'", "")
    msg = weights + " missing"
    r = 1
    if len(weights) > 0 and not os.path.isfile(weights):
        d = {""}
        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        if not (
            r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1000000.0
        ):
            os.remove(weights) if os.path.exists(weights) else None
            s = ""
            r = os.system(s)
            if not (
                r == 0
                and os.path.exists(weights)
                and os.path.getsize(weights) > 1000000.0
            ):
                os.remove(weights) if os.path.exists(weights) else None
                raise Exception(msg)


def gdrive_download(id="1n_oKgR81BJtqk75b00eAjdv03qVCQn2f", name="coco128.zip"):
    t = time.time()
    print(
        "Downloading https://drive.google.com/uc?export=download&id=%s as %s... "
        % (id, name),
        end="",
    )
    os.remove(name) if os.path.exists(name) else None
    os.remove("cookie") if os.path.exists("cookie") else None
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(
        'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id=%s" > %s '
        % (id, out)
    )
    if os.path.exists("cookie"):
        s = (
            'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm=%s&id=%s" -o %s'
            % (get_token(), id, name)
        )
    else:
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    r = os.system(s)
    os.remove("cookie") if os.path.exists("cookie") else None
    if r != 0:
        os.remove(name) if os.path.exists(name) else None
        print("Download error ")
        return r
    if name.endswith(".zip"):
        print("unzipping... ", end="")
        os.system("unzip -q %s" % name)
        os.remove(name)
    print("Done (%.1fs)" % (time.time() - t))
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""
