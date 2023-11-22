# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image


from torchmetrics.classification import Dice

from monai.metrics import DiceHelper, DiceMetric


import utils

from .template import SimpleToolTemplate
from .metrics import get_confusion_matrix

# TODO: 고쳐야됨 Test 코드 및 Visualization 코드


class Segmentation(SimpleToolTemplate):
    def __init__(
        self,
        model: nn.Module,
        optim_name: str,
        device: torch.device,
        distributed: bool,
        output_dir: str,
        print_freq: int,
        **kwargs,
    ) -> None:
        super().__init__(model, device, distributed, output_dir, print_freq)
        self.model = model
        self.num_classes = model.num_classes

        self.set_optimizer(optim_name, **kwargs)
        # TODO: set_lr_scheduler()

    def train_one_epoch(self, epoch, train_loader, wandb):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))

        dice_metric = Dice(num_classes=self.num_classes, ignore_index=0).to(self.device)

        header = f"Epoch: [{epoch}]"
        step = 0
        for step, (inputs, labels) in enumerate(
            metric_logger.log_every(train_loader, self.print_freq, header), 1
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model.forward_with_losses(inputs, labels)
            logits = outputs["logits"]
            loss = outputs["loss"]

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            preds = torch.softmax(logits, dim=1).argmax(dim=1)
            dice_metric.update(preds, labels)

            metric_logger.update(
                loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"]
            )
            wandb.log(
                data={"Loss/train": loss.item()}, step=len(train_loader) * epoch + step
            )

        dice_score = dice_metric.compute().item()
        print(f"{header} Dice score: {dice_score:.4f} \n")
        train_loss = metric_logger.meters["loss"].global_avg
        wandb.log(
            data={
                "Dice/train": dice_score,
                "Loss/avgTrain": train_loss,
                "LR": metric_logger.meters["lr"].global_avg,
            },
            step=step * (1 + epoch),
        )
        return train_loss

    def validation(self, global_step, valid_loader, wandb):
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Validation"

        dice_metric = Dice(num_classes=self.num_classes, ignore_index=0).to(self.device)

        with torch.inference_mode():
            for inputs, labels in metric_logger.log_every(
                valid_loader, self.print_freq, header
            ):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model.forward_with_losses(inputs, labels)
                logits = outputs["logits"]
                loss = outputs["loss"]

                metric_logger.update(loss=loss.item())

                preds = torch.softmax(logits, dim=1).argmax(dim=1)
                dice_metric.update(preds, labels)

            metric_logger.synchronize_between_processes()

            dice_score = dice_metric.compute().item()
        val_loss = metric_logger.meters["loss"].global_avg
        print(f"{header} Dice score: {dice_score:.4f}")
        wandb.log(
            data={
                "Dice/valid": dice_score,
                "Loss/avgValid": val_loss,
            },
            step=global_step,
        )
        return {"Dice/valid": dice_score}, val_loss

    def train(
        self,
        ith_fold,
        wandb,
        start_epoch,
        epochs,
        train_loader,
        valid_loader,
        train_sampler,
        args,
    ):
        best_metric_dict = {}
        fit_val_loss = 99999
        for epoch in range(start_epoch, epochs):
            print("-" * 15, f" {ith_fold} Fold ", "-" * 15)
            if self.distributed:
                train_sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(epoch, train_loader, wandb)

            # TODO: self.lr_scheduler.step()

            global_step = (epoch + 1) * len(train_loader)

            metric_dict, valid_loss = self.validation(global_step, valid_loader, wandb)

            if self.distributed:
                model_without_ddp = self.model.modules
            else:
                model_without_ddp = self.model

            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": valid_loss,
                "args": args,
                "ith_fold": ith_fold,
            }

            utils.save_on_master(
                checkpoint,
                os.path.join(self.output_dir, f"fold{ith_fold}", f"ckpt.pth"),
            )
            if best_metric_dict == {}:
                best_metric_dict = metric_dict.copy()

            if fit_val_loss > valid_loss:
                fit_val_loss = valid_loss
                save_path = os.path.join(
                    self.output_dir, f"fold{ith_fold}", f"fit_val_loss.pth"
                )
                utils.save_on_master(checkpoint, save_path)

            for metric_name, metric_value in best_metric_dict.items():
                if metric_value <= metric_dict[metric_name]:
                    best_metric_dict[metric_name] = metric_dict[metric_name]
                    metric_name = metric_name.replace("/", "_")
                    save_path = os.path.join(
                        self.output_dir, f"fold{ith_fold}", f"best_{metric_name}.pth"
                    )
                    utils.save_on_master(checkpoint, save_path)

    @staticmethod
    def test(model, test_dataset, device, test_output_dir, batch_size=4):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        num_classes = test_dataset.num_classes
        classes = test_dataset.classes
        model.eval()

        micro_dice_metric = Dice(
            num_classes=num_classes, ignore_index=0, average="micro"
        ).to(device)
        macro_dice_metric = Dice(
            num_classes=num_classes, ignore_index=0, average="macro"
        ).to(device)
        each_dice_metric = DiceHelper(
            include_background=False,
            reduction="none",
            sigmoid=True,
            softmax=True,
            get_not_nans=False,
        )

        file_name_list = []
        false_positives_list = []
        false_negatives_list = []

        metric_dict = {f"{cls}": [] for cls in classes.keys() if cls != "background"}

        batch_size = test_loader.batch_size

        with torch.inference_mode():
            for inputs, labels, file_names in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(inputs)

                preds = torch.softmax(logits, dim=1)

                micro_dice_metric.update(preds, labels)
                macro_dice_metric.update(preds, labels)
                batch_dice_score = each_dice_metric(preds, labels.unsqueeze(1))

                for i in range(batch_size):
                    file_name_list.append(file_names[i])
                    score = batch_dice_score[i]
                    for cls, index in classes.items():
                        if cls == "background":
                            continue
                        metric_dict[cls].append(score[index - 1].item())

                    pred_idx = torch.unique(preds.argmax(dim=1)[i])
                    label_idx = torch.unique(labels[i])

                    # pred 에는 있지만, Label에는 없는 것: 잘 못 예측함
                    fp_idx = torch.nonzero(torch.isin(pred_idx, label_idx, invert=True)).squeeze()
                    fp_idx = fp_idx.cpu().detach().numpy().tolist()
                    if not isinstance(fp_idx, int):
                        fp_idx = ",".join(map(str, fp_idx))
                    false_positives_list.append(fp_idx)
                    # pred 에는 없지만, Label에는 있는 것: Missing 놓친 것
                    fn_idx = torch.nonzero(torch.isin(label_idx, pred_idx, invert=True)).squeeze()
                    fn_idx = fn_idx.cpu().detach().numpy().tolist()
                    if not isinstance(fn_idx, int):
                        fn_idx = ",".join(map(str, fn_idx))
                    false_negatives_list.append(fn_idx)

            micro_dice_score = micro_dice_metric.compute().item()
            macro_dice_score = macro_dice_metric.compute().item()

        print(f"Micro Dice: {micro_dice_score:.4f}")
        print(f"Macro Dice: {macro_dice_score:.4f}")

        temp_table = {"file_name": file_name_list}
        temp_table["false_positives"] = false_positives_list
        temp_table["false_negatives"] = false_negatives_list

        for cls in classes.keys():
            if cls == "background":
                continue
            temp_table[cls] = metric_dict[cls]

        df = pd.DataFrame(temp_table, columns=temp_table.keys())
        df.to_csv(os.path.join(test_output_dir, "result.csv"))
        with open(os.path.join(test_output_dir, "result.tex"), "w") as f:
            f.write(df.to_latex(index=False))
        with open(os.path.join(test_output_dir, "result.md"), "w") as f:
            f.write(df.to_markdown(index=False))

    @staticmethod
    def inference(model: nn.Module, image: torch.Tensor):
        model.eval()
        logit = model(image.unsqueeze(0))
        pred_prob = torch.softmax(logit, dim=1)
        pred = pred_prob.argmax(dim=1)
        return {"pred_prob": pred_prob, "pred": pred}

    @staticmethod
    def visualization(model: nn.Module, test_dataset, device, test_output_dir, batch_size=4):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        num_classes = test_dataset.num_classes
        classes = test_dataset.classes
        inv_classes = test_dataset.inv_classes
        model.eval()

        micro_dice_metric = Dice(
            num_classes=num_classes, ignore_index=0, average="micro"
        ).to(device)
        each_dice_metric = DiceHelper(
            include_background=False,
            reduction="none",
            sigmoid=True,
            softmax=True,
            get_not_nans=False,
        )

        batch_size = test_loader.batch_size

        with torch.inference_mode():
            for inputs, labels, file_names in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(inputs)

                preds_prob = torch.softmax(logits, dim=1)
                preds = torch.argmax(preds_prob, dim=1)
                micro_dice_metric.update(preds_prob, labels)
                batch_dice_score = each_dice_metric(preds_prob, labels.unsqueeze(1))
                
                inputs = inputs.cpu()
                labels = labels.cpu()
                preds = preds.cpu()

                for i in range(batch_size):
                    fig, axs = plt.subplots(3, num_classes, figsize=(24,18))
                    fig.subplots_adjust(wspace=0.5, hspace=0.5)
                    if inputs.size(1) == 1:
                        img = torch.cat([inputs[i], inputs[i],inputs[i]],dim=0) # FIXME: 임시 방편
                    else:
                        img = inputs[i]
                    img = ((img - img.max()) / (img.max() - img.min()) *255.0).type(torch.uint8)
                    mask = torch.nn.functional.one_hot(labels[i], num_classes).type(torch.bool).permute(2,0,1)
                    pred_mask = torch.nn.functional.one_hot(preds[i], num_classes).type(torch.bool).permute(2,0,1)
                    axs[0][0].imshow(img.permute(1,2,0), cmap="gray")
                    axs[0][0].set_title(f"Image {file_names[i]}")
            
                    axs[1][0].imshow(draw_segmentation_masks(img, mask, alpha=0.8).permute(1,2,0))
                    axs[1][0].set_title("Target Mask")

                    axs[2][0].imshow(draw_segmentation_masks(img, pred_mask, alpha=0.8).permute(1,2,0))
                    axs[2][0].set_title("prediction Mask")

                    pred_idx = torch.unique(preds[i])
                    label_idx = torch.unique(labels[i])
                    # pred 에는 있지만, Label에는 없는 것: 잘 못 예측함
                    fp_idx = torch.nonzero(torch.isin(pred_idx, label_idx, invert=True)).squeeze()
                    # pred 에는 없지만, Label에는 있는 것: Missing 놓친 것
                    fn_idx = torch.nonzero(torch.isin(label_idx, pred_idx, invert=True)).squeeze()

                    for j in range(1, num_classes):
                        temp_label_view = torch.zeros(labels[i].size())
                        temp_label_view[labels[i] == j] = 1
                        temp_pred_view = torch.zeros(labels[i].size())
                        temp_pred_view[preds[i] == j] = 1

                        axs[0][j].imshow(temp_label_view)
                        axs[1][j].imshow(temp_pred_view)
                        dice_score = batch_dice_score[i]
                        axs[0][j].set_title(f"{inv_classes[j]} | {dice_score[j-1]:.4f}")
                        if dice_score[j-1] > 0:
                            axs[1][j].set_title(f"Prediction")
                        else:# FIXME: 여기 고쳐야됨...
                            if "999434_M" in file_names[i]:
                                print(file_names[i], fp_idx, fn_idx, pred_idx, label_idx)
                            if "1115560_M" in file_names[i]:
                                print(file_names[i], fp_idx, fn_idx, pred_idx, label_idx)
                            if j in fp_idx:
                                axs[1][j].set_title(f"Prediction : Error - False Positive")
                            elif j in fn_idx:
                                axs[1][j].set_title(f"Prediction : Error - Missing")
                            else:
                                axs[1][j].set_title(f"Prediction")

                        axs[2][j].imshow(temp_label_view*temp_pred_view)
                        axs[2][j].set_title("overlap")
                    plt.tight_layout()
                    plt.savefig(os.path.join(test_output_dir, f"{file_names[i]}.png"))
                    plt.close()
