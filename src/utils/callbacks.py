import matplotlib.pyplot as plt
import torch

from lightning.pytorch import Callback, Trainer, LightningModule
from torch.optim import Optimizer

from src.models.pitch_module import PitchModule

from lightning.pytorch.utilities import grad_norm


class LogGradNorms(Callback):
    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        # log only generator grads
        grad_norms = grad_norm(pl_module, norm_type='inf')
        # grad_norms = {k: v for k, v in grad_norms.items() if 'generator' in k or 'total' in k}
        pl_module.log_dict(grad_norms, on_step=True)

def plot_predict_vs_truth(y_hat, y_gt):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(y_hat.cpu().numpy()[0], label='y_hat')
    plt.plot(y_gt.cpu().numpy()[0], label='y_gt')
    plt.legend()
    return fig

class LogResults(Callback):
    def on_fit_start(self, trainer: Trainer, pl_module: PitchModule) -> None:
        speech, pitch, probability,*_ = trainer.datamodule.train_dataloader().dataset[0]
        pl_module.example_input_array = speech.unsqueeze(0)
        self.x = pl_module.transfer_batch_to_device(speech.unsqueeze(0), pl_module.device, 0) 
        self.y = pl_module.transfer_batch_to_device(pitch.unsqueeze(0), pl_module.device, 0)
        self.y_prob = pl_module.transfer_batch_to_device(probability.unsqueeze(0), pl_module.device, 0)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: PitchModule) -> None:
        with torch.no_grad():
            y_hat = pl_module.predict_step((self.x, self.y, self.y_prob), 0)
            # decode the gt as well
            y_gt = pl_module.decoder.idx_pitch[self.y.argmax(dim=-1)]
            fig = plot_predict_vs_truth(y_hat, y_gt)
            trainer.logger.experiment.add_figure('predict_vs_truth', fig, global_step=trainer.current_epoch)
            