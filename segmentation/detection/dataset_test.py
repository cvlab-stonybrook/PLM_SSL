import json
import os

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


chest_labels = ["Atelectasis", "Calcification", "Consolidation", "Effusion", "Emphysema", "Fibrosis", "Fracture",
                "Mass", "Nodule", "Pneumothorax"]
chest_labels_dic = {l:i for i, l in enumerate(chest_labels)}


def get_chest_dicts(img_dir="datasets/ChestX/", set='train'):
    json_file = os.path.join(img_dir, "annotations", "%s.json" % set)
    with open(json_file) as f:
        imgs_anns = json.load(f)
    # print(imgs_anns[0])
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}

        filename = os.path.join(img_dir, set, v["file_name"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        boxes = v["boxes"]
        labels = v["syms"]
        objs = []
        for box, label_str in zip(boxes, labels):
            print(type(box), box)
            obj = {
                "bbox": box,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": chest_labels_dic[label_str],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for d in ["train", "test"]:
    DatasetCatalog.register("ChestX_" + d, lambda d=d: get_chest_dicts(set=d))
    MetadataCatalog.get("ChestX_" + d).set(thing_classes=chest_labels)
balloon_metadata = MetadataCatalog.get("ChestX_train")

if __name__ == '__main__':
    get_chest_dicts()