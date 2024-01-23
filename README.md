# ANFIS_EnergyEfficience

## Introduction

This exercise presents an exploration into the efficiency of Adaptive Neuro-Fuzzy Inference Systems (ANFIS) in predicting the heating load of residential buildings—a key aspect in enhancing energy efficiency.  

Through testing of different ANFIS configurations, we aim to identify optimal parameters that align with the predictive accuracy presented in Tsanas and Xifara's1 paper. By comparing several model iterations and evaluating them based on Mean Absolute Error (MAE), we seek to establish a streamlined, yet accurate, ANFIS model for Heat Loss prediction. The exercise encourages a blend of theoretical understanding and practical application, fostering a hands-on approach to learning in the domain of Computational Intelligence. 

Results are compared to

[1] Tsanas, A., & Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. Energy and Buildings, 49, 560-567. https://doi.org/10.1016/j.enbuild.2012.03.003 

## Setup

- Matlab R2022b
- [Parallel Computing Toolbox](https://es.mathworks.com/products/parallel-computing.html?s_tid=AO_PR_info)

## Usage

- Perform a grid search to find the best ANFIS configuration for the dataset:
- - `train_anfis.m`
- Make 100 trains with different splits with the best ANFIS configuration:
- - `train_best_anfis.m`