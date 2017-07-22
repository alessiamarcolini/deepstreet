# Deepstreet

> Deepstreet is the project I developed for my high school thesis in IT @ ITI Marconi, Verona (IT).
This project aims to provide a system able to recognize the type of a street sign into an image, using Deep Learning techniques.

## Features

- Written in Python
- Keras as a main library for Deep Learning, with Numpy and OpenCV
- Pre-trained weights available


## Getting Started

### Prerequisistes
To run all the scripts you need the following packages:
- Python version 3.5
- `numpy` v. 1.13
- `matplotlib` v. 2.0
- `OpenCV` v. 3.0
- `tensorflow` v. 1.1 -- or `tensorflow-gpu` if you have GPUs on your machine
- `keras` v. 2.0
- `hdf5` v. 1.8 and `h5py` v. 2.7

Optional, but recommended:
- **NVIDIA cuDNN** if you have NVIDIA GPUs on your machine.
    [https://developer.nvidia.com/rdp/cudnn-download]()




The easiest way to get (most) these is to use an all-in-one installer such as [Anaconda](http://www.continuum.io/downloads) from Continuum. This is available for multiple architectures.

### Running scripts
To run the scripts, just download this repo and execute:
```shell
python <filename.py>
```
The two main scripts (`deepstreet_training.py` and `deepstreet_predict.py`) can be executed with optional arguments. To find out the arguments for, let's say, `deepstreet_training.py` execute:

```shell
python deepstreet_training.py --help
```


## Contributing Guidelines

All contributions and suggestions are welcome!

For suggested improvements, please create an issue.


## License

This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/alessiamarcolini/deepstreet/blob/master/LICENSE.txt) file for details
