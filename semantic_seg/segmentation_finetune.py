import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim import lr_scheduler
from torchmetrics import Dice, JaccardIndex

try:
    from torchmetrics import F1
except Exception as e:
    from torchmetrics import F1Score as F1


class SegmentationFinetuneModule(pl.LightningModule):
    def __init__(self, model, loss, args, img_fn=None, mask_fn=None):
        super(SegmentationFinetuneModule, self).__init__()

        self.args = args
        self.model = model

        self.loss = loss

        self.lr = args.lr

        self.save_hyperparameters("args")

        self.micro_f1 = F1(task='multiclass', num_classes=args.num_classes, average="micro")
        self.dice = Dice(num_classes=args.num_classes, average="micro")
        self.jaccard = JaccardIndex(task='multiclass', num_classes=args.num_classes, average="micro")

        self.dice_macro = Dice(num_classes=args.num_classes, average="macro")
        self.jaccard_macro = JaccardIndex(task='multiclass', num_classes=args.num_classes, average="macro")

        self.img_fn = img_fn
        self.mask_fn = mask_fn


    def forward(self, x, mask=None):
        # print(x.shape, label.shape)

        mask_pred = self.model(x)
        if isinstance(mask_pred, (list, tuple)):
            mask_pred = mask_pred[0]
        if mask is None:
            return mask_pred

        mask = mask.flatten(0)
        loss_area = mask > 0
        mask = mask[loss_area]

        mask_pred = mask_pred.flatten(2).transpose(1, 2)
        mask_pred = mask_pred.flatten(0, 1)[loss_area, :]

        loss = self.loss(mask_pred, mask)
        return loss, mask_pred, mask

    def forward_step(self, batch, stage):
        batch, _ = batch
        mask = batch[-1]
        x = batch[:-1]
        if len(x) == 1:
            x = x[0]

        if self.img_fn is not None:
            x = self.img_fn(x)
        if self.mask_fn is not None:
            mask = self.mask_fn(mask)

        loss, mask_pred, mask_loss = self.forward(x, mask)

        self.log('%s_loss' % stage, loss, on_step=True, on_epoch=True, sync_dist=True)

        f1 = self.micro_f1(mask_pred, mask_loss)
        self.log('%s_F1' % stage, f1, on_step=False, on_epoch=True, sync_dist=True)

        dice = self.dice(mask_pred, mask_loss)
        self.log('%s_Dice' % stage, dice, on_step=False, on_epoch=True, sync_dist=True)

        dice_macro = self.dice_macro(mask_pred, mask_loss)
        self.log('%s_Dice_macro' % stage, dice_macro, on_step=False, on_epoch=True, sync_dist=True)

        jaccard = self.jaccard(mask_pred, mask_loss)
        self.log('%s_Jaccard' % stage, jaccard, on_step=False, on_epoch=True, sync_dist=True)

        jaccard_macro = self.jaccard_macro(mask_pred, mask_loss)
        self.log('%s_Jaccard_macro' % stage, jaccard_macro, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.forward_step(batch, "val")

    def configure_optimizers(self):
        parameters = self.parameters()
        # if self.args.weight_decay is None:
        #     cus_optimizer = torch.optim.AdamW(parameters, lr=self.lr)
        # else:
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs, eta_min=0)
        # cus_sch = lr_scheduler.MultiStepLR(cus_optimizer, self.args.decay_multi_epochs, gamma=self.args.decay_rate,
        #                               last_epoch=-1, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler":  {
                # REQUIRED: The scheduler instance
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }