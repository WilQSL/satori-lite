"""
Engine configuration module - wraps neuron config for engine use
"""
import os


def root(*args):
    """Returns the engine root path"""
    return os.path.join('/Satori/Engine', *args)


def dataPath(*args):
    """Returns the data storage path"""
    return os.path.join('/Satori/data', *args)


def modelPath(*args):
    """Returns the model storage path"""
    return os.path.join('/Satori/models', *args)
