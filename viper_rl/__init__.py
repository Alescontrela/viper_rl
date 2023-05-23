import pathlib

VIPER_PATH = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = VIPER_PATH / 'configs'
CHECKPOINT_PATH = VIPER_PATH.parent / 'viper_rl_data' / 'checkpoints'

directory = pathlib.Path(__file__).resolve()
__package__ = directory.name
