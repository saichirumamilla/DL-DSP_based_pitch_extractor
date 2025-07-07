from typing import Any, Dict, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import numpy as np

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.encoders import Encoder
from src.models.components.extractors import FeatureExtractor
from src.models.components.opus_extractors import FeatureExtractor
from src.models.components.decoders import Decoder
from src.models.components.metrics import RCAMetric

class PitchModule(pl.LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """
    def __init__(
        self,
        extractor: FeatureExtractor,
        encoder: Encoder,
        decoder: Callable[...,Decoder],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criteria: torch.nn.Module,
        compile: bool = True,
        external_data_evaluate: pl.LightningDataModule | None = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.extractor = extractor
        self.encoder = encoder
        self.decoder = decoder(input_size=self.encoder.output_size)
        
        # loss function
        self.criterion= torch.nn.CrossEntropyLoss()
        self.criteria = criteria

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.decoder.output_size) # get classes from decoder as property
        self.val_acc = Accuracy(task="multiclass", num_classes=self.decoder.output_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.decoder.output_size)

        self.train_rca = RCAMetric()
        self.val_rca = RCAMetric()
        self.test_rca = RCAMetric(test_mode=True)
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_rca_best = MaxMetric()
        self.test_rca_best = MaxMetric()

        self.train_rca_values = []
        self.val_rca_values = []
        self.test_rca_values = []
        self.eps= 1e-5

        self.external_data_evaluate = external_data_evaluate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of features
        :return: A tensor of pitch values.
        """
        ext = self.extractor(x)
        enc = self.encoder(ext)
        pitch_probs = self.decoder(enc)
        return pitch_probs
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor `x`.

        :param x: A tensor of input features.
        :return: A tensor of extracted features.
        """
        ext = self.extractor(x)
        return ext
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> Any:
        if self.external_data_evaluate is not None:
            batch = self.external_data_evaluate()
        y_pred, y_pitch, y_prob = self.model_step(batch)
        # max likelihood prediction
        idx = y_pred.argmax(dim=-1) #batch_size, time_frame, num_freq_bins
        return self.decoder.idx_pitch[idx]

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_rca.reset()
        self.val_rca_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        
        x, y_pitch, y_prob,*rmse = batch
        y_pred = self.forward(x)
        y_pred = y_pred[:, :y_pitch.size(1), :]
        mask =  (y_prob > 0)[...,None].expand_as(y_pred)
        y_pred = y_pred * mask
        y_pitch = y_pitch * mask
        return y_pred, y_pitch, y_prob

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # forward pass        
        y_pred, y_pitch, y_prob = self.model_step(batch)
        # update and log metrics
        loss = self.criteria(input=y_pred.transpose(-1, -2), target=y_pitch.transpose(-1, -2), reduction='none')
        loss = torch.mean(loss * y_prob)
        self.train_loss(loss)
        # update and log metrics
        preds = y_pred.argmax(dim=-1)
        gt = y_pitch.argmax(dim=-1)
        self.train_acc(preds, gt)
        self.train_rca.update(self.decoder.idx_pitch[preds], self.decoder.idx_pitch[gt], y_prob)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rca", self.train_rca, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail

        # if self.current_epoch == 78 and batch_idx == 56:
        #     print("Loss is", loss)
            #import IPython; IPython.embed()
        return loss    

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        train_rca_value = self.train_rca.compute()  # get current train acc
        self.train_rca_values.append(train_rca_value.item())
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.no_grad():
            y_pred, y_pitch, y_prob = self.model_step(batch)
        # update and log metrics
        loss = self.criteria(input=y_pred.transpose(-1, -2), target=y_pitch.transpose(-1, -2), reduction='none')
        loss = torch.mean(loss * y_prob)
        self.val_loss(loss)

        preds = y_pred.argmax(dim=-1)
        gt = y_pitch.argmax(dim=-1)
        self.val_acc(preds, gt)

        self.val_rca.update(self.decoder.idx_pitch[preds], self.decoder.idx_pitch[gt], y_prob)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)  
        self.log("val/rca", self.val_rca, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        rca = self.val_rca.compute()  # get current val acc
        self.val_rca_best(rca)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.val_rca_values.append(rca.item())
        self.log("val/acc_best", self.val_rca_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        with torch.no_grad():
            y_pred, y_pitch, y_prob = self.model_step(batch)
        loss = self.criteria(input=y_pred.transpose(-1, -2), target=y_pitch.transpose(-1, -2), reduction='none') 
        loss = torch.mean(loss * y_prob) # Weighted loss
        
        self.test_loss(loss) # log loss
        preds = y_pred.argmax(dim=-1) 
        gt = y_pitch.argmax(dim=-1)  
        self.test_acc(preds, gt)
        self.test_rca.update(self.decoder.idx_pitch[preds], self.decoder.idx_pitch[gt], y_prob)
       
        # update and log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rca", self.test_rca, on_step=False, on_epoch=True, prog_bar=True)
        rca_value = self.test_rca.compute()
        rca_path= 'results/rca_per_utt_amazon_fda.csv'
        with open(rca_path, 'a') as f:
            np.savetxt(f, [rca_value.cpu()], delimiter=",", fmt='%.6f')

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        #rca_per_utterance = self.test_rca.compute()  # get current val acc
        rca_per_utterance = self.test_rca.ret_per_utterance_rca()
        log = self.logger.log_dir
        #np.savetxt(f"{log}/rca_per_utt.csv", rca_per_utterance.numpy(), delimiter=",")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile:
            self.extractor = torch.compile(self.extractor) 
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
       
        optimizer = self.hparams.optimizer 
        optimizer= optimizer(params=self.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/rca",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PitchModule()
