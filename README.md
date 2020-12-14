# Critical Assessment of Artificial Intelligence Methods for Prediction of hERG Channel Inhibition in the “Big Data” Era

This repository contains the code and data related to our [recent article](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00884){:target="_blank"} that compares classical ML appproaches with newer AI techniques in predicting hERG channel inhibition. A breif description of contents of the repository is provided below.

* `data/`       :  datasets used to build and validate the models
* `ext_models/` :  prospective validation results from external hERG models (StarDrop and PredhERG)
* `notebooks/`  :  a jupyter notebook (and model dependecies) that allows consensus prediction on a test dataset
* `scripts/`    :  scripts used to build hERG models


## Predicting a new test set

The Jupyter notebook in `notebooks/` can be used to perform a consenus prediction based on the best individual models developed in this study.


### Instructions

The code requires two types of molecular descriptors to be calculated before hand: RDKit descriptors and Morgan fingerprints. The models were built using RDKit features (a total of 119 descriptors) and Morgan fingerprints (1024 bits; radius 2) that were calculated in KNIME. The first column of the file must be SMILES followed by the RDKit descriptors and Morgan fingerprints in the same order. An example test set is available in the code: `notebooks/blockers_sampled.csv`.
 

1. Clone the repository and fetch all files (some files are large and need to be fetched using git lfs)

    ```
    git clone https://github.com/ncats/herg-ml.git
    cd herg-ml
    git lfs install
    git lfs fetch
    git lfs pull
    
    ```
    
2. Create and activate a conda environment

    ```
    conda create -n herg-ml python=3.6
    conda activate herg-ml
    bash install.sh
    ```
    
3. Launch Jupyter (opens the default web browser - http://localhost:8888/tree)

    ```
    jupyter notebook
    ```
   
4. Open the file `notebooks/consensus_model.ipynb` and execute the notebook following the in-line instructions

5. To end the notebook session, press `Ctrl+C` and choose `y` when prompted to shutdown the notebook server

6. Deactive the conda environment when finshed

    ```
    conda deactivate
    ```

**Note**: These instructions were tested in MacOS with Python 3.6


## Results

We compared our consensus model (using the [prospective validation set](https://github.com/ncats/herg-ml/blob/master/data/train_valid/validation_set.csv)) against previous hERG models proposed by [Braga et al.](https://pubmed.ncbi.nlm.nih.gov/24805060/) ([Pred-hERG 4.2](http://predherg.labmol.com.br/)) and [Ryu et al.](https://academic.oup.com/bioinformatics/article/36/10/3049/5727757) ([DeepHIT](https://academic.oup.com/bioinformatics/article/36/10/3049/5727757)).

**Model** | **Balanced Accuracy** | **Sensitivity** | **Specificity** |
| :---: | :---: | :---: | :---: |
Our Consensus | 0.80 | 0.74 | 0.86 |
Pred-hERG 4.2<sup>a</sup> | 0.77 | 0.74 | 0.81 |
DeepHIT | 0.75 | 0.73 | 0.77 |

<sup>a</sup> Pred-hERG 4.2 returned predictions for 835 out 839 validation set compounds.

## Web-based Predictions

A web-based prediction service will be made available in future. If you experience troubles using the currently available models, please [contact us](mailto:vishalbabu.siramshetty@nih.gov?subject=[GitHub]%20hERG%20Models).
