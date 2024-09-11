# LFCC - K-Fold Adaptation

This repository is an adaptation of the [ASVspoof 2021 LFCC-LCNN Baseline](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-LFCC-LCNN), with modifications to support k-fold cross-validation for model training and inference.


### Initial Setup

Before running any scripts, ensure you have the required environment configured. You can install dependencies by using the provided `conda.sh` script.

### Configuration

- **Scripts to Modify**: You need to configure the `config.py` file to specify paths to dataset. Additionally, check the variables in each of the scripts (`00_train.sh`, `01_eval.sh`, `02_metric_scores.sh`) before running them to adjust them to your preferred directory.
- **Adjustable Parameters**: You can modify training parameters such as the number of epochs, batch size, and learning rates directly in the scripts to suit your specific needs.

### Scripts Overview

1. **00_train.sh**:  
   This script performs both training and inference using k-fold cross-validation. The results of the inference process are included in the same logs as the training output.

2. **01_eval.sh**:  
   This script evaluates a trained model using the test dataset specified in `config.py`.

3. **02_metric_scores.sh**:  
   After evaluating the model, use this script to calculate the metric scores. It processes the files generated during evaluation (those with a `_scores` suffix) and outputs the relevant metrics.

General usage for the scripts above are as follows:
    
```bash
bash 00_train.sh
```

For more information from the original source code, check the inner README inside each folder (LA/PA)