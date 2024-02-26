# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from torchmetrics.classification import (
    ConfusionMatrix,
    AUROC,
    ROC,
    F1Score,
    Accuracy,
    PrecisionRecallCurve,
    Precision,
    Recall,
    AveragePrecision,
    Specificity,
)
from torchmetrics.utilities.plot import plot_curve
from torchmetrics.utilities.compute import _auc_compute_without_check
from torchmetrics.functional.classification.auroc import _reduce_auroc

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

import utils
from .regularization import orthognal_l2_reg
from .template import SimpleToolTemplate


def add_argparser_cls_tool(task: str, parent_parser: ArgumentParser, optim_name: str) -> ArgumentParser:
    if task == "binary":
        parent_parser = BinaryClassification.add_argparser_optim(parent_parser, optim_name)
        parent_parser = BinaryClassification.add_argparser_criterion(parent_parser)
        return parent_parser
    elif task == "multiclass":
        parent_parser = MultiClassification.add_argparser_optim(parent_parser, optim_name)
        parent_parser = MultiClassification.add_argparser_criterion(parent_parser)
        return parent_parser
    else:
        raise NotImplementedError(f"'{task}' tool is not supported.")


def create_cls_tool(
    task: str,
    model: nn.Module,
    num_classes: int,
    optim_name: str,
    criterion_name: str,
    device: torch.device,
    distributed: bool,
    output_dir: str,
    print_freq: int,
    use_orthogonal_l2_reg: bool,
    **kwargs,
):
    if task == "binary":
        return BinaryClassification(
            model,
            optim_name,
            criterion_name,
            device,
            distributed,
            output_dir,
            print_freq,
            use_orthogonal_l2_reg,
            **kwargs,
        )
    elif task == "multiclass":
        return MultiClassification(
            model,
            num_classes,
            optim_name,
            criterion_name,
            device,
            distributed,
            output_dir,
            print_freq,
            use_orthogonal_l2_reg,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"'{task}' tool is not supported.")


class BinaryClassification(SimpleToolTemplate):
    def __init__(
        self,
        model: nn.Module,
        optim_name: str,
        criterion_name: str,
        device: torch.device,
        distributed: bool,
        output_dir: str,
        print_freq: int,
        use_orthogonal_l2_reg: bool,
        **kwargs,
    ) -> None:
        super().__init__(model, device, distributed, output_dir, print_freq)
        self.model = model
        self.num_classes = 2

        self.set_optimizer(optim_name, **kwargs)
        self.set_criterion(criterion_name, **kwargs)
        self.use_orthogonal_l2_reg = use_orthogonal_l2_reg

    def set_criterion(self, criterion_name: str, **kwargs):
        if criterion_name == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"{criterion_name} is not supported")

    def add_argparser_criterion(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Criterion")
        parser.add_argument("--criterion-name", required=True, type=str, help="")
        return parent_parser

    def train_one_epoch(self, epoch, train_loader, wandb):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))

        f1 = F1Score(task="binary").to(self.device)
        acc = Accuracy(task="binary").to(self.device)

        header = f"Epoch: [{epoch}]"
        step = 0
        for step, (inputs, targets, _) in enumerate(metric_logger.log_every(train_loader, self.print_freq, header), 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)

            loss = self.criterion(logits.squeeze(dim=1), targets.float())
            if self.use_orthogonal_l2_reg:
                loss += orthognal_l2_reg(self.model, device=self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prob_preds = torch.sigmoid(logits.squeeze(dim=1))
            f1.update(prob_preds, targets)
            acc.update(prob_preds, targets)

            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            wandb.log(data={"Loss/train": loss.item()}, step=len(train_loader) * epoch + step)

        accuracy = acc.compute().item()
        f1_score = f1.compute().item()
        print(f"{header} Accuracy: {accuracy:.4f} | F1Score: {f1_score:.4f}\n")
        train_loss = metric_logger.meters["loss"].global_avg
        wandb.log(
            data={
                "Accuracy/train": accuracy,
                "F1score/train": f1_score,
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

        f1 = F1Score(task="binary").to(self.device)
        acc = Accuracy(task="binary").to(self.device)

        with torch.inference_mode():
            for inputs, targets, _ in metric_logger.log_every(valid_loader, self.print_freq, header):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = self.model(inputs)

                loss = self.criterion(logits.squeeze(dim=1), targets.float())
                if self.use_orthogonal_l2_reg:
                    loss += orthognal_l2_reg(self.model, device=self.device)

                prob_preds = torch.sigmoid(logits.squeeze(dim=1))
                f1.update(prob_preds, targets)
                acc.update(prob_preds, targets)

                metric_logger.update(loss=loss.item())

            metric_logger.synchronize_between_processes()

            accuracy = acc.compute().item()
            f1_score = f1.compute().item()
        val_loss = metric_logger.meters["loss"].global_avg
        print(f"{header} Accuracy: {accuracy:.4f} | F1Score: {f1_score:.4f}")
        wandb.log(
            data={
                "Accuracy/valid": accuracy,
                "F1score/valid": f1_score,
                "Loss/avgValid": val_loss,
            },
            step=global_step,
        )
        return {"Accuracy/valid": acc, "F1Score/valid": f1}, val_loss

    @staticmethod
    def test(model, test_dataset, device, test_output_dir, batch_size=4):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()

        confusion_matrix = ConfusionMatrix(task="binary").to(device)
        precision = Precision(task="binary").to(device)
        recall = Recall(task="binary").to(device)
        spc = Specificity(task="binary").to(device)
        f1 = F1Score(task="binary").to(device)
        acc = Accuracy(task="binary").to(device)

        roc = ROC(task="binary").to(device)
        auroc = AUROC(task="binary").to(device)

        prc = PrecisionRecallCurve(task="binary")
        ap = AveragePrecision(task="binary")

        metrics = [confusion_matrix, precision, recall, spc, f1, acc, roc, auroc, prc, ap]

        with torch.inference_mode():
            for inputs, labels, _ in tqdm(test_loader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(inputs)
                prob_preds = torch.sigmoid(logits.squeeze(dim=1))

                for _m in metrics:
                    _m.update(prob_preds, labels)

            roc.compute()  # For plot the Receiver Operating Characteristic (ROC).
            precision_list, recall_list, threshold_list = prc.compute()
            precision_list = precision_list.cpu()
            recall_list = recall_list.cpu()
            threshold_list = threshold_list.cpu()
            auroc_score = auroc.compute().cpu().item()
            ap_score = ap.compute().cpu().item()
            auprc_score = _auc_compute_without_check(recall_list, precision_list, -1.0).item()

            confusion_matrix.compute().cpu()  # For plot the Confusion Matrix.
            precision_score = precision.compute().cpu().item()
            recall_score = recall.compute().cpu().item()
            spc_score = spc.compute().cpu().item()
            f1_score = f1.compute().cpu().item()
            accuracy = acc.compute().cpu().item()

        print(f"AUROC: {auroc_score:.4f}")
        print(f"Precision: {precision_score:.4f} | Recall: {recall_score:.4f} | Specificity: {spc_score:.4f}")
        print(f"Accuracy: {accuracy:.4f} | F1 Score: {f1_score:.4f}")
        print(f"AUPRC: {auprc_score:.4f} | mAP: {ap_score:.4f}")

        if test_output_dir is not None:
            confusion_matrix.plot()
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(test_output_dir, "ConfusionMatrix.png"), dpi=300)
            plt.close()

            roc.plot(score=True)
            plt.title("Receiver Operating Characteristic Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(test_output_dir, "ROC.png"), dpi=300)
            plt.close()

            # prc.plot(score=True)
            plot_curve(
                (recall_list, precision_list, threshold_list),
                score=_auc_compute_without_check(recall_list, precision_list, -1.0),
                label_names=("Recall", "Precision"),
            )
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                plt.ylim([0.0, 1.05])
                plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
            plt.title("Precision Recall Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(test_output_dir, "PRC.png"), dpi=300)
            plt.close()

            prc.plot(score=True)
            plt.title("Recall Precision Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(test_output_dir, "RPC.png"), dpi=300)
            plt.close()

        result_data = {
            "F1 Score": f1_score,
            "Precisoin": precision_score,
            "Recall": recall_score,
            "Specificity": spc_score,
            "Accuracy": accuracy,
            "AUROC": auroc_score,
            "AUPRC": auprc_score,
        }

        return result_data

    @staticmethod
    def visualization(model: nn.Module, test_dataset, device, test_output_dir: str, batch_size: int = 1):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        save_path = os.path.join(test_output_dir, "visual")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for inputs, targets, filenames in test_loader:
            inputs = inputs.to(device, non_blocking=True)

            logits = model(inputs)

            preds_prob = torch.sigmoid(logits)
            preds = preds_prob.clone().cpu()
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

            for i in range(batch_size):
                image = inputs[i].cpu()
                target_name = test_dataset.inv_classes[targets[i].item()]
                pred_name = test_dataset.inv_classes[preds[i].item()]
                image = to_pil_image(image)
                plt.imshow(image)
                plt.title(f"{filenames[i]} | Target: {target_name} | Pred: {pred_name}")
                plt.savefig(os.path.join(save_path, f"{filenames[i]}.png"))
                plt.close()

    @staticmethod
    def visualization_with_cam(
        model: nn.Module, input_image: torch.Tensor, target_layer, num_samples: int = 4, std=0.3
    ):
        model.eval()

        input_shape = input_image.shape
        cam_extractor = SmoothGradCAMpp(model, target_layer, num_samples, std=std, input_shape=input_shape)

        logits = model(input_image.unsqueeze(0))
        prob_pred = torch.sigmoid(logits)

        activation_map = cam_extractor(logits.squeeze(0).argmax().item(), logits)

        img = to_pil_image(input_image.cpu())
        act_map = to_pil_image(activation_map[0].squeeze(0), mode="F")
        result = overlay_mask(img, act_map, alpha=0.5)

        return result, prob_pred


class MultiClassification(SimpleToolTemplate):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optim_name: str,
        criterion_name: str,
        device: torch.device,
        distributed: bool,
        output_dir: str,
        print_freq: int,
        use_orthogonal_l2_reg: bool,
        **kwargs,
    ) -> None:
        super().__init__(model, device, distributed, output_dir, print_freq)
        self.model = model
        self.num_classes = num_classes

        self.set_optimizer(optim_name, **kwargs)
        self.set_criterion(criterion_name, **kwargs)
        self.use_orthogonal_l2_reg = use_orthogonal_l2_reg

    def set_criterion(self, criterion_name: str, **kwargs):
        if criterion_name == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{criterion_name} is not supported")

    def add_argparser_criterion(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Criterion")
        parser.add_argument("--criterion-name", required=True, type=str, help="")
        return parent_parser

    def train_one_epoch(self, epoch, train_loader, wandb):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))

        f1 = F1Score(task="multiclass", num_classes=self.num_classes).to(self.device)
        acc = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)

        header = f"Epoch: [{epoch}]"
        step = 0
        for step, (inputs, targets, _) in enumerate(metric_logger.log_every(train_loader, self.print_freq, header), 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)

            loss = self.criterion(logits, targets)
            if self.use_orthogonal_l2_reg:
                loss += orthognal_l2_reg(self.model, device=self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prob_preds = torch.softmax(logits, dim=1)
            f1.update(prob_preds, targets)
            acc.update(prob_preds, targets)

            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            wandb.log(data={"Loss/train": loss.item()}, step=len(train_loader) * epoch + step)

        accuracy = acc.compute().item()
        f1_score = f1.compute().item()
        print(f"{header} Accuracy: {accuracy:.4f} | F1Score: {f1_score:.4f}\n")
        train_loss = metric_logger.meters["loss"].global_avg
        wandb.log(
            data={
                "Accuracy/train": accuracy,
                "F1score/train": f1_score,
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

        f1 = F1Score(task="multiclass", num_classes=self.num_classes).to(self.device)
        acc = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)

        with torch.inference_mode():
            for inputs, targets, _ in metric_logger.log_every(valid_loader, self.print_freq, header):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = self.model(inputs)

                loss = self.criterion(logits, targets)
                if self.use_orthogonal_l2_reg:
                    loss += orthognal_l2_reg(self.model, device=self.device)

                prob_preds = torch.softmax(logits, dim=1)

                f1.update(prob_preds, targets)
                acc.update(prob_preds, targets)

                metric_logger.update(loss=loss.item())

            metric_logger.synchronize_between_processes()

            accuracy = acc.compute().item()
            f1_score = f1.compute().item()
        val_loss = metric_logger.meters["loss"].global_avg
        print(f"{header} Accuracy: {accuracy:.4f} | F1Score: {f1_score:.4f}")
        wandb.log(
            data={
                "Accuracy/valid": accuracy,
                "F1score/valid": f1_score,
                "Loss/avgValid": val_loss,
            },
            step=global_step,
        )
        return {"Accuracy/valid": acc, "F1Score/valid": f1}, val_loss

    @staticmethod
    def test(model, test_dataset, device, test_output_dir, batch_size=4):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        num_classes = test_dataset.num_classes
        model.eval()
        task = "binary" if num_classes == 2 else "multiclass"
        confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes).to(device)
        precision = Precision(task=task, num_classes=num_classes, average="macro").to(device)
        recall = Recall(task=task, num_classes=num_classes, average="macro").to(device)
        spc = Specificity(task=task, num_classes=num_classes, average="macro").to(device)
        f1 = F1Score(task=task, num_classes=num_classes, average="macro").to(device)
        acc = Accuracy(task=task, num_classes=num_classes, average="macro").to(device)

        roc = ROC(task=task, num_classes=num_classes).to(device)
        auroc = AUROC(task=task, num_classes=num_classes, average="macro").to(device)

        prc = PrecisionRecallCurve(task=task, num_classes=num_classes)
        ap = AveragePrecision(task=task, num_classes=num_classes)

        metrics_for_pred = [confusion_matrix, precision, recall, spc, f1, acc]
        metrics_for_prob_pred = [roc, auroc, prc, ap]

        with torch.inference_mode():
            for inputs, targets, _ in tqdm(test_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(inputs)
                prob_preds = torch.softmax(logits, dim=1)
                preds = prob_preds.argmax(dim=1)

                for _m in metrics_for_pred:
                    _m.update(preds, targets)

                for _m in metrics_for_prob_pred:
                    if task =="binary":
                        temp_prob_preds = prob_preds[:, 1]
                    else:
                        temp_prob_preds = prob_preds
                    _m.update(temp_prob_preds, targets)

            roc.compute()  # For plot the Receiver Operating Characteristic (ROC).
            precision_list, recall_list, threshold_list = prc.compute()
            auroc_score = auroc.compute().cpu().item()
            ap_score = ap.compute().cpu().item()
            if task == "binary":
                auprc_score = _auc_compute_without_check(recall_list, precision_list, -1.0).item()
            else:
                auprc_score = - _reduce_auroc(recall_list, precision_list).item()
            cm = confusion_matrix.compute().cpu()  # For plot the Confusion Matrix.
            precision_score = precision.compute().cpu().item()
            recall_score = recall.compute().cpu().item()
            spc_score = spc.compute().cpu().item()
            f1_score = f1.compute().cpu().item()
            accuracy = acc.compute().cpu().item()
            
        print("="*30)
        print(f"Confusion Matrix\n{cm}")
        print(f"AUROC: {auroc_score:.4f}")
        print(f"Precision: {precision_score:.4f} | Recall: {recall_score:.4f} | Specificity: {spc_score:.4f}")
        print(f"Accuracy: {accuracy:.4f} | F1 Score: {f1_score:.4f}")
        print(f"AUPRC: {auprc_score:.4f} | mAP: {ap_score:.4f}\n")
        print("="*30)

        result_data = {
            "F1 Score": f1_score,
            "Precisoin": precision_score,
            "Recall": recall_score,
            "Specificity": spc_score,
            "Accuracy": accuracy,
            "AUROC": auroc_score,
            "AUPRC": auprc_score,
        }
        if test_output_dir is None:
            return result_data
        else:
            confusion_matrix.plot()
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(test_output_dir, "ConfusionMatrix.png"), dpi=300)
            plt.close()

            roc.plot(score=True)
            plt.title("Receiver Operating Characteristic Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(test_output_dir, "ROC.png"), dpi=300)
            plt.close()

            # FIXME: Torchmetric에서 버그가 Fix 되면 prc 변수를 사용하면 됨.
            if task == "binary":
                score = _auc_compute_without_check(recall_list, precision_list, -1.0)
            else:
                score = -1.0 * _reduce_auroc(recall_list, precision_list, average="none")
            plot_curve(
                (recall_list, precision_list, threshold_list),
                # FIXME: torchmetric 에서 Precision Recall curve 의 AUC 계산 값이 마이너스가 나옴...
                score = score,  
                label_names=("Recall", "Precision"),
            )
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                plt.ylim([0.0, 1.05])
                plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
            plt.title("Precision Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precison")
            plt.tight_layout()
            plt.savefig(os.path.join(test_output_dir, "PRC.png"), dpi=300)
            plt.close()

            prc.plot(score=True)
            plt.title("Precision Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precison")
            plt.tight_layout()
            plt.savefig(os.path.join(test_output_dir, "PRC_legacy.png"), dpi=300)
            plt.close()
            return result_data

        

    @staticmethod
    def visualization(model, test_dataset, device, test_output_dir, batch_size=1):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        save_path = os.path.join(test_output_dir, "visual")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for inputs, targets, filenames in tqdm(test_loader):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)

            preds_prob = torch.softmax(logits, dim=1)
            preds = torch.argmax(preds_prob, dim=1)

            for i in range(batch_size):
                image = inputs[i].cpu()
                target_name = test_dataset.inv_classes[targets[i].item()]
                pred_name = test_dataset.inv_classes[preds[i].item()]
                image = to_pil_image(image)
                plt.imshow(image)
                plt.title(f"{filenames[i]} | Target: {target_name} | Pred: {pred_name}")
                plt.savefig(os.path.join(save_path, f"{filenames[i]}.png"))
                plt.close()

    @staticmethod
    def visualization_with_cam(
        model: nn.Module, input_image: torch.Tensor, target_layer, num_samples: int = 4, std: float = 0.3
    ):
        model.eval()

        input_shape = input_image.shape
        cam_extractor = SmoothGradCAMpp(model, target_layer, num_samples, std=std, input_shape=input_shape)

        logits = model(input_image.unsqueeze(0))
        prob_pred = torch.softmax(logits, dim=1)

        activation_map = cam_extractor(logits.squeeze(0).argmax().item(), logits)

        img = to_pil_image(input_image.cpu())
        act_map = to_pil_image(activation_map[0].squeeze(0), mode="F")
        result = overlay_mask(img, act_map, alpha=0.5)

        return result, prob_pred
