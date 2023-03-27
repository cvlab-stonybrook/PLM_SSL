import os
from PIL import Image
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

src_dir = r"/gpfs/scratch/jingwezhang/data/Xray/NIH/original/images"
dst_dir = r"/dev/shm/jzhang/NIH"

scale_percent = 25

postfix = ''
# postfix = '_org'
in_extension = '.png'
out_extension = '.png'

ignore_if_exist = False

n_jobs = 20


def resize_img(file_name):
    if ignore_if_exist and os.path.isfile(os.path.join(dst_dir, file_name)):
        return
    img = cv2.imread(os.path.join(src_dir, file_name), cv2.IMREAD_UNCHANGED)
    if len(img.shape) > 2:
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise NotImplementedError

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    s_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    assert len(s_img.shape) == 2

    save_file_name = file_name[:-len(in_extension)] + out_extension
    cv2.imwrite(os.path.join(dst_dir, save_file_name), s_img)
    # print(s_img.shape)


def get_shape(file_name):
    img = cv2.imread(os.path.join(src_dir, file_name), cv2.IMREAD_UNCHANGED)
    if len(img.shape) > 2:
        print(file_name)
        return True
    return False


def main():
    img_fns = [f for f in os.listdir(src_dir) if f.lower().endswith(postfix+in_extension)]
    print("Converting totally %d images." % len(img_fns))
    os.makedirs(dst_dir, exist_ok=True)
    Parallel(n_jobs=n_jobs, backend='loky')(delayed(resize_img)(img_fn) for img_fn in tqdm(img_fns))
    # for img_fn in img_fns:
    #     # resize_img(img_fn)
    #     get_shape((img_fn))


if __name__ == '__main__':
    main()
    print("Done")