import pathlib

VIPER_DATA_PATH = pathlib.Path(__file__).resolve().parent
VIPER_DATASET_PATH = pathlib.Path(__file__).resolve().parent / 'datasets'
VIPER_CHECKPOINT_PATH = pathlib.Path(__file__).resolve().parent / 'checkpoints'

directory = pathlib.Path(__file__).resolve()
__package__ = directory.name
