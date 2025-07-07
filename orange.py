
import sys

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchaudio
import lightning.pytorch as pl
from scipy.stats import gaussian_kde
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np


from IPython.display import Audio, display

from pathlib import Path
from egaznepy.visualize import apply_plot_style
from src.data.mocha_timit_datamodule import MTIMITDataModule
from src.models.pitch_module import PitchModule
from src.models.components.metrics import RCAMetric
from torchmetrics.classification.accuracy import Accuracy



apply_plot_style(0.9)
plt.rcParams['text.usetex'] = False

#root_dir = Path(globals()['_dh'][0]).parent
#print(root_dir)
import os
#os.chdir(root_dir)

#config_path_predict = 'logs/debug/runs/2024-05-29_17-03-54/.hydra/config.yaml'
config_path_predict = 'weights/joint/2024-05-29_17-03-54/.hydra/config.yaml'
#ckpt_path = 'logs/debug/runs/2024-05-29_17-03-54/checkpoints/epoch_135.ckpt'
ckpt_path = 'weights/joint/2024-05-29_17-03-54/checkpoints/epoch_135.ckpt'

#config_path_eval = 'logs/eval/runs/2024-06-02_17-08-12/.hydra/config.yaml'
config_path_eval = 'logs/eval/runs/yaapt_model/.hydra/config.yaml'




config_predict = OmegaConf.load(config_path_predict)
datamodule = hydra.utils.instantiate(config_predict.data, slice_length=4)
pitch_module: PitchModule = hydra.utils.instantiate(config_predict.model)

config_eval = OmegaConf.load(config_path_eval)
datamodule = hydra.utils.instantiate(config_eval.data, slice_length=4)
yaapt_module: PitchModule = hydra.utils.instantiate(config_eval.model)

del config_predict.trainer.default_root_dir
del config_eval.trainer.default_root_dir
config_predict.trainer.limit_predict_batches = 1

trainer = hydra.utils.instantiate(config_predict.trainer) 
arr_predict = trainer.predict(model=pitch_module, datamodule=datamodule, ckpt_path=ckpt_path)
arr_eval = trainer.predict(model=yaapt_module, datamodule=datamodule)


# Extract the first 4 elements - our model

pitches_predicted = arr_predict[0][0:4]
pitches_eval = arr_eval[0][0:4]
# gt
corresponding_data = next(iter(datamodule.test_dataloader()))
corresponding_audio = corresponding_data[0][0:4]
pitches_gt = corresponding_data[1][0:4]
probs_gt = corresponding_data[2][0:4]
mask_gt = probs_gt > 0
pitches_gt = pitch_module.decoder.idx_pitch[pitches_gt.argmax(dim=-1)]
#pitches_gt = pitches_gt * mask_gt
pitches_predicted =pitches_predicted * mask_gt
pitches_predicted = torch.where(pitches_predicted == 0, torch.tensor(62.5), pitches_predicted)

print("pitches_shape",pitches_gt.shape)
print("probs_shape",probs_gt.shape)



rca = RCAMetric(test_mode=False)
out = rca.ret_current_rca(pitches_predicted, pitches_eval, probs_gt)

print("out",out)