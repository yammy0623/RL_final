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

Thresholds of sparse reward are set 0.75, 0.89, 0.93 for T of 5, 10, 20.
The setting of thresholds impacts a lot. For example, experiment of T=5 with threshold of 0.8 only gets FID of 66.33.

### CIFAR 10 
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDIM |68.28|20.76|11.46|5.71|
| RL   |34.67|18.13|11.02|-|

### LSUN-Church
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDIM |  |19.10 | | |
| RL   |  | | |-|