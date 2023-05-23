# Video Prediction Models as Rewards for Reinforcement Learning

Code for VIPER (Video Predcition Rewards), a general algorithm which leverages video prediction models as priors for Reinforcement Learning.

![VIPER Tasks](https://github.com/Alescontrela/VIPER/assets/13845012/f7f30610-4985-409e-bfe5-e82c125cca7a)

If you found this code useful, please reference it in your paper:

```
@article{escontrela2023viper,
  title={Video Prediction Models as Rewards for Reinforcement Learning},
  author={Alejandro Escontrela and Ademi Adeniji and Wilson Yan and Ajay Jain and Xue Bin Peng and Ken Goldberg and Youngwoon Lee and Danijar Hafner and Pieter Abbeel},
  journal={arXiv preprint arXiv:2305.14343},
  year={2023}
}
```

For more information:
- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## VIPER 🐍

VIPER leverages the next-frame log likelihoods of a pre-trained video prediction model as rewards for downstream reinforcement learning tasks. The method is flexible to the particular choice of video prediction model and reinforcement learning algorithm. The general method outline is shown below:

![viper_method_figure](https://github.com/Alescontrela/VIPER/assets/13845012/5f258fce-8793-442c-9e80-3f0ebb4df741)

This code release provides a reference VIPER implementation which uses [VideoGPT](https://arxiv.org/abs/2104.10157) as the video prediction model and [DreamerV3](https://arxiv.org/abs/2301.04104) as the reinforcement learning agorithm.

## Install:

Create a conda environment with Python 3.8:

```
conda create -n viper python=3.8
conda activate viper
```

Install [Jax][jax].

Install dependencies:
```
pip install -r requirements.txt
```

## Downloading Data

Download the DeepMind Control Suite expert dataset with the following command:

```
python -m viper_rl_data.download dataset dmc
```

and the Atari dataset with:

```
python -m viper_rl_data.download dataset atari
```

This will produce datasets in `<VIPER_INSTALL_PATH>/viper_rl_data/datasets/` which are used for training the video prediction model. The location of the datasets can be retrieved via the `viper_rl_data.VIPER_DATASET_PATH` variable.

## Downloading Checkpoints

Download the DeepMind Control Suite videogpt/vqgan checkpoints with:

```
python -m viper_rl_data.download checkpoint dmc
```

and the Atari checkpoint with:

```
python -m viper_rl_data.download checkpoint atari
```

This will produce video model checkpoints in `<VIPER_INSTALL_PATH>/viper_rl_data/checkpoints/`, which are used downstream for RL. The location of the checkpoints can be retrieved via the `viper_rl_data.VIPER_CHECKPOINT_PATH` variable.

## Video Model Training

Use the following command to first train a VQ-GAN:
```
python scripts/train_vqgan.py -o viper_rl_data/checkpoints/dmc_vqgan -c viper_rl/configs/vqgan/dmc.yaml
```

To train the VideoGPT, update `ae_ckpt` in `viper_rl/configs/dmc.yaml` to point to the VQGAN checkpoint, and then run:
```
python scripts/train_videogpt.py -o viper_rl_data/checkpoints/dmc_videogpt_l16_s1 -c viper_rl/configs/videogpt/dmc.yaml
```

## Policy training

Checkpoints for various models can be found in `viper_rl/videogpt/reward_models/__init__.py`. To use one of these video models during policy optimization, simply specify it with the `--reward_model` argument.  e.g.

```
python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_cartpole_balance --reward_model=dmc_clen16_fskip4 --logdir=~/logdir
```

Custom checkpoint directories can be specified with the `$VIPER_CHECKPOINT_DIR` environment variable. The default checkpoint path is set to `viper_rl_data/checkpoints/`.

**Note**: For Atari, you will need to install [atari-py][ataripy] and follow the Atari 2600 VCS ROM install instructions.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier
[paper]: https://arxiv.org/pdf/2305.14343.pdf
[website]: https://escontrela.me/viper
[tweet]: https://twitter.com/AleEscontrela/status/1661363555495710721?s=20
[ataripy]: https://github.com/openai/atari-py