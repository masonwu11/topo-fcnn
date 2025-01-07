# Functional Connectomes of Neural Networks

## About

This repository written in Python contains implementations of clustering connectomes of neural networks. It serves as the code appendix of the AAAI 2025 paper:
- Functional Connectomes of Neural Networks

Please cite our paper if you use this code in your research:
```
@inproceedings{songdechakraiwutwu2024functional,
  title={Functional connectomes of neural networks},
  author={Songdechakraiwut, Tananun and Wu, Yutong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

There are mainly two types of files in the directory: **pipeline** and **job** files. Additionally, a `runtime` folder under the main directory contains implementations for running runtime experiments of all methods.

## Environment Setup
It is recommended to use a virtual environment like conda to manage packages. The following packages are recommeneded to be installed before running the scripts in this repository:

- **NumPy**
- **Pandas**
- **SciPy**
- **Matplotlib**
- **Scikit-learn**
- **Scikit-learn-extra**
- **Torch**
- **Torchvision**
- **Ripser**
- **Persim**

## Directory Structure

### `runtime` Folder
This folder contains the scripts related to running runtime experiments. **`runtime_analysis.py`** contains the main script for running runtime experiments with different sizes of networks and different distances/kernels.

### Pipeline Files
Pipeline files define methods that are used in jobs. These files don't execute jobs directly but provide the necessary methods that are utilized in job files. **`loading_pipeline.py`** loads different datasets and performs the preprocessing steps required before training tasks. **`training_pipeline.py`** further provides methods for hyperparameter tuning, model training, and functional connectome generation. **`clustering_pipeline.py`** contains methods related to topological clustering of connectomes for (Top/Adj). **`traditional_clustering_pipeline.py`** contains the traditional clustering methods serving as baseline, such as k-medroids with bottleneck distance (BD), Wasserstein distance (WD), sliced Wasserstein kernel (SWK), heat kernel (HK), and k-means with persistence images (PI).

### Job Files
Job files execute specific tasks using the methods defined in the pipeline files. Running these files often requires the name of dataset (mnist/fashion-mnist/cifar10) as a command line input to perform various operations like loading, training, or clustering related to the dataset. **`training_job.py`**, **`clustering_job.py`**, and **`traditional_clustering_job.py`** correspond to executing methods in pipeline files of the same names.

### Other Files
- **`models.py`** describes the architectures of three models used in the experiments with three datasets. 
- **`top_clustering.py`** contains topological clustering methods that are employed in the methods Top and Adj. Many methods are utilized in **`clustering_pipeline.py`**.

    #### Reference
    Songdechakraiwut, Tananun, Bryan M. Krause, Matthew I. Banks, Kirill V. Nourski, and Barry D. Van Veen.  
    "Fast topological clustering with Wasserstein distance."  
    *International Conference on Learning Representations (ICLR)*, 2022.

