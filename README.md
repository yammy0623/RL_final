## Quickstart

Install required packages first:
```bash
pip install -r requirements.txt
```

Please download the CIFAR10 model via wget and put it at `model/ddpm_ema_cifar10`:
```bash
wget https://github.com/VainF/Diff-Pruning/releases/download/v0.0.1/ddpm_ema_cifar10.zip
```
PS: Note this model is only supported by old diffusers.

To train RL, run:
```python
python train.py
```

To evaluate FID, run:
```bash
bash eval_fid.sh
```
PS. Please modify `--save_path` in eval_fid.sh, this means the path of the RL model produced by `train.py`.

(Note all these codes may not be 100% accurate)

## Experimental Results

### CIFAR 10 
| Step | DDIM | RL |
| --- | --- | --- |
| 5 |  |
| 10 | 20.76 | 18.13
| 20 | |
| 100|  5.71 | - 
