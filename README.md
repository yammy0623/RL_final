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
