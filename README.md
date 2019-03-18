# Anomaly detection in a network
*Authors : Marin BOUTHEMY and Nicolas TOUSSAINT*

This library has been made to run the anomaly detection designed by [Elliott & al.](https://arxiv.org/pdf/1901.00402.pdf) on a networkX instance. You can find the full description of their work [here](https://arxiv.org/pdf/1901.00402.pdf).

Their module is divided in 5 sub-modules. Each module compute different kind of features for a network. It uses a set of global features for the full network and community features for partition of the network.

![alt text](https://github.com/Marin35/Anomaly-detection-ENSAE/raw/master/docs/schema.png)

*Credits : this illustration of anomalies and of the different modules was found in the paper of [Elliott & al.](https://arxiv.org/pdf/1901.00402.pdf)*

## Requirements
The library has some requirements :
 - Python 3
 - Numpy
 - Pandas
 - python-louvain (available [here](https://github.com/taynaud/python-louvain))
 - networkX

To install all this requirement you can use the requirements.txt.

## Installation
To use our library, you just need to import its main folder to your Python. You can do it like that :
```python
import sys
sys.path.insert(0, "anomaly_detection/") # Path of the library folder on your computer
```

## Documentation
To understand and see the results of the library, you can look at the two examples in the folder "examples".  

 1. [Results](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/examples/Results.ipynb) : use the features available in the features folder to fit a RandomForest classifier and look at its performance.
 2. [Features creation](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/examples/Features%20generation.ipynb) : generate a network and build its features

## Files structure
The library contains a lot of files, however each file is made for a specific module of the paper and has been commented.

First the files for each section of the paper :
 - [GAW.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/GAW.py) -> compute the first global features of the module (section 3.1)
 - [communities.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/communities.py) -> compute the communities features (section 3.2)
 - [localisation.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/localisation.py) -> compute the eigenvectors statistics on the augmented statistics (section 3.3)
 - [path_finder.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/path_finder.py) -> Compute the path features (section 3.4)
 - [net_emd.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/net_emd.py) -> compute the NetEMD features (section 3.5). It uses the statistics defined in [set_statistics.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/set_statistics.py)

More general files :
 - [anomalies.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/anomalies.py) -> a set of function to add anomalies to a network
 - [features.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/features.py) -> the main file to generate features for a network. It basically merged all the modules together.
 - [generation.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/generation.py) -> a set of function to generate a network and also generate configuration model. Also provide function to generate a Monte Carlo distribution of a statistics with null configuration of a network.
 - [heavy_path.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/heavy_path.py) -> a set of function to do the augmentation step of your network.
 - [utils.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/anomaly_detection/utils.py) -> a set of useful function for all the modules.
 - [generator.py](https://github.com/Marin35/Anomaly-detection-ENSAE/blob/master/generator.py) -> useful to generate a lot of network and their features.
