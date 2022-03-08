# DP-Ensemble: Diversity Optimized Ensemble
-----------------
[![GitHub license](https://img.shields.io/badge/license-apache-green.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![Version](https://img.shields.io/badge/version-0.0.1-red.svg?style=flat)]()
<!---
[![Travis Status]()]()
[![Jenkins Status]()]()
[![Coverage Status]()]()
--->

## Introduction

DP-Ensemble is short for **D**iversity o**P**timized **Ensemble**, which is built on top of [EnsembleBench](https://github.com/git-disl/EnsembleBench). By leveraging FQ-diversity metrics, DP-Ensemble can effectively identify high diversity ensembles with high performance. 

FQ-diversity metrics are designed based on the following three optimizations:
1. separately measure and compare the ensemble teams of equal size.
2. leverage the negative samples from the focal model to measure ensemble diversity.
3. partition the candidate ensemble teams by using binary clustering with strategically selected initial centroids.

These optimizations enable FQ-diversity metrics to more accurately capture the failure independence among the member models of ensemble teams, and efficiently select high quality ensemble teams. Furthermore, the quality of selected ensemble teams can be improved by introducing EQ diversity metrics to combine the top performing FQ metrics.

CVPR 2021 Presentation Video: https://youtu.be/jmHTCE3mrR4

If you find this work useful in your research, please cite the following paper:

**Bibtex**:
```bibtex
@INPROCEEDINGS{dp-ensemble,
    author={Wu, Yanzhao and Liu, Ling and Xie, Zhongwei and Chow, Ka-Ho and Wei, Wenqi},
    booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    title={Boosting Ensemble Accuracy by Revisiting Ensemble Diversity Metrics}, 
    year={2021},
    volume={},
    number={},
    pages={16464-16472},
    doi={10.1109/CVPR46437.2021.01620}
}
```

## Instructions
 
Following the steps below for using our FQ metrics for selecting high quality ensemble teams.

1. Obtain the pretrained models for the dataset <dataset> according to the model files under <dataset> folder.
2. Extract the prediction vectors and labels for <dataset> and store them under <dataset>/prediction for testing data and <dataset>/train for training data.
3. Execute the FQEnsembleSelection.py file to obtain the results.

Please refer to our paper and supplementary for detailed results.

You can check a simplified version for the focal diversity based ensemble selection here: https://github.com/git-disl/EnsembleBench/blob/main/demo/FocalDiversityBasedEnsembleSelection.ipynb

## Problem


## Installation
    pip install -r requirements.txt

## Supported Platforms


## Development / Contributing


## Issues


## Status


## Contributors

See the [people page](https://github.com/git-disl/DP-Ensemble/graphs/contributors) for the full listing of contributors.

## License

Copyright (c) 20XX-20XX [Georgia Tech DiSL](https://github.com/git-disl)  
Licensed under the [Apache License](LICENSE).

