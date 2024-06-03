import os
from torch.utils.data import Dataset
import random
import json
import io
import imageio.v2 as imageio
import base64
import numpy as np
from PIL import Image, ImageDraw
LA_THRESHOLD = 0.7
def lid(text, model):
    span_start, span_end = 0, len(text)
    det_text = text[span_start:span_end]
    res = model.predict(det_text)
    la = res[0][0].replace("__label__", "")
    prob = res[1][0]
    return la, prob

class CDIPBase(Dataset):
    def __init__(
        self,
        size=256,
        mount_root: str = "/home/yilinjia/mycontainer/",
        source_relative_path: str = "tengchao/dataset/kosmos_d/ocr_line/train.json",
        data_dir_relative_path: str = "tengchao/dataset/kosmos_d/ocr_line/",  # and /parallel
    ):
        self.size = size
        source_path = os.path.join(mount_root, source_relative_path)
        assert os.path.exists(source_path), f"cdip source path {source_path} not exists"

        self.data_dir = os.path.join(mount_root, data_dir_relative_path)
        # self.receipt_dir = os.path.join(mount_root, "dataset")
        assert os.path.exists(
            self.data_dir
        ), f"cdip data dir {self.data_dir} not exists"

        # Source files
        self.source = json.load(open(source_path, "r"))
        self.weight = [i["weight"] for i in self.source]
        # make the weight of "receipts" to be 0 this dataset is dirty
        # self.weight = [0.5 if i == "receipts" else i for i in self.weight]
        self.length = [len(i["source"]) - 1 for i in self.source]
        # define the whole dataset as an iterator
        self.inf_iter = self.setup_iterator()

    def __len__(self):
        return 9999999999

    def setup_iterator(self):
        # Create an infinite iterator
        def inf_shuffle_generator():
            while True:
                i = random.choices(range(len(self.source)), weights=self.weight)[0]
                entry = self.source[i]
                j = random.randint(0, self.length[i])
                source_file = entry["source"][j]
                iterator = self.read_file(source_file)
                try:
                    while True:
                        yield next(iterator)
                except StopIteration:
                    continue

        return inf_shuffle_generator()

    def __getitem__(self, i):
        return next(self.inf_iter)

    def read_file(self, file_relative_path: str):
        """
        This is a generator
        """
        try: 
            file_path = os.path.join(self.data_dir, file_relative_path)
            if "parallel" in file_relative_path:
                # replace ocr_line with parallel
                file_path = file_path.replace("ocr_line", "parallel")
            if not os.path.exists(file_path):
                print("| file {} not exists".format(file_path), flush=True)
                return iter([])
            # entries is a list of dict contains ['image', 'lines', 'bboxs']
            with open(file_path, "r", encoding="utf8") as f:
                entries = f.read().strip().split("\n")
            # shuffle the entries
            random.shuffle(entries)
            for entry_s in entries:
                # entry_s is a string, we need convert it to json
                try:
                    assert len(entry_s.strip()) != 0
                    entry = json.loads(entry_s.strip())
                    if "boxes" in entry:
                        entry["bboxs"] = entry["boxes"]
                        del entry["boxes"]
                    assert len(entry["bboxs"]) == len(
                        entry["lines"]
                    ), f"bboxs and lines not match: {entry['bboxs']} {entry['lines']}"
                    pic = Image.open(
                        io.BytesIO(base64.b64decode(entry["image"]))
                    ).convert("RGB")
                    extrema = pic.convert("L").getextrema()
                    assert extrema[0] != extrema[1], "image is blank"
                    del pic
                    pic = imageio.imread(
                        io.BytesIO(base64.b64decode(entry["image"])), pilmode="RGB"
                    )
                    original_width, original_height, _ = pic.shape
                    # print(original_width, original_height)
                    assert original_width * 8 >= original_height, "img too high"
                    assert original_height * 8 >= original_width, "img too wide"
                    # crop the image and resize to size
                    min_size = min(original_width, original_height)
                    if min_size > 1.4 * self.size:
                        min_size = int(1.4*self.size)
                    # print(min_size, self.size)
                    # crop from center
                    x1 = (original_width - min_size) // 2
                    y1 = (original_height - min_size) // 2
                    x2 = x1 + min_size
                    y2 = y1 + min_size
                    image = pic[y1:y2, x1:x2]
                    image = Image.fromarray(image)
                    # print(image.size)
                    image = image.resize((self.size, self.size))
                    image = np.array(image).astype(np.uint8)
                    image = (image / 127.5 - 1.0).astype(np.float32)
                    # transpose
                    # image = np.transpose(image, (0, 2, 1))
                    # print("image numpy shape",image.shape)
                    yield {
                        "image": image,
                    }
                except Exception as e:
                    continue

        except Exception as e:
            return iter([])
        # finish reading all the entries in the file
        return iter([])

class CDIPTrain(CDIPBase):
    def __init__(self, size):
        super().__init__()
    def __len__(self):
        return 999999999

class CDIPTest(CDIPBase):
    def __init__(self, size):
        super().__init__()
    def __len__(self):
        return 64

if __name__ == "__main__":
    # test the CDIPBase
    cdip = CDIPBase(size=512)
    print(len(cdip))
    # save cdip[0]['image'] to a file
    img = cdip[0]['image']
    img = (img + 1.0) * 127.5
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    print(img.size)
    img.save("test.png")