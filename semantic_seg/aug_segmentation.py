class DataAugmentationImgMask(object):
    def __init__(self, augmentation, img_fn=None, mask_fn=None):
        self.augmentation = augmentation
        self.mask_fn = mask_fn
        self.img_fn = img_fn

    def __call__(self, x):
        img, mask = x
        x = self.augmentation(image=img, mask=mask)
        img, mask = x["image"], x["mask"]
        if self.mask_fn is not None:
            mask = self.mask_fn(mask)
        if self.img_fn is not None:
            img = self.img_fn(img)
        if isinstance(img, (list, tuple)):
            return [*img, mask]
        else:
            return img, mask

class DataAugmentationImgMask(object):
    def __init__(self, augmentation, img_fn=None, mask_fn=None):
        self.augmentation = augmentation
        self.mask_fn = mask_fn
        self.img_fn = img_fn

    def __call__(self, x):
        img, mask = x
        x = self.augmentation(image=img, mask=mask)
        img, mask = x["image"], x["mask"]
        if self.mask_fn is not None:
            mask = self.mask_fn(mask)
        if self.img_fn is not None:
            img = self.img_fn(img)
        if isinstance(img, (list, tuple)):
            return [*img, mask]
        else:
            return img, mask
