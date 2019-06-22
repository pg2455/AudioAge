import logging, json
import numpy as np
from scipy import signal
import soundfile as sf
import torch

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

def parse_soundfile(filepath, timeframe=5, window_fn=signal.gaussian(50, std=1)):
    """
    timeframe: The length of audio in seconds to consider
    window_fn: The characteristic of the window
    """
    input_length = timeframe * 16000
    data, fs = sf.read(filepath)
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset: (input_length + offset)]
    else:
        max_offset = input_length - len(data)
        offset = np.random.randint(max_offset)
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = audio_norm(data)
    # print(data.max(), data.min())
    # sd.play(data, fs)
    # sd.wait()

    f, t, Sxx = signal.spectrogram(data, window=window_fn)
    # print(f.shape, t.shape, Sxx.shape, data.shape, fs)
    return torch.Tensor(Sxx).unsqueeze(0)



class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


#https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: [%(filename)s:%(lineno)d] %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: [%(filename)s:%(lineno)d] %(message)s'))
        logger.addHandler(stream_handler)
