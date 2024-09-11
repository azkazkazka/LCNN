#!/bin/bash
########################
# Script for evaluation
# Usage:
#   1. please check that config.py has been properly configured
#   2. $: bash 00_train.sh > /dev/null 2>&1 &
#   3. please check log_train and log_err to monitor the training 
#      process
#   
# Note:
#   1. The script by default uses the pre-trained model from ASVspoof2019
#      If you don't want to use it, just delete the option --trained-model
#   2. For options, check $: python main.py --help
########################

log_train_name=log_fold_train_and_infer
log_err_name=log_err
epoch_name=epoch
network_name=trained_network

echo -e "Training"
echo -e "Please monitor the log training: $PWD/${log_train_name}.txt\n"
source $PWD/../../env.sh
python main.py --model-forward-with-file-name \
       --num-workers 3 --epochs 50 \
       --batch-size 16 \
       --sampler block_shuffle_by_length \
       --save-trained-name ${network_name} \
       --save-epoch-name ${epoch_name} \
       --lr-decay-factor 0.5 --lr-scheduler-type 1 \
       --ignore-training-history-in-trained-model \
       --lr 0.001 --seed 1000 > ${log_train_name}.txt 2>${log_err_name}.txt
echo -e "Training process finished"
echo -e "Training log has been written to $PWD/${log_train_name}.txt"
echo -e "Training in progress log has been done"
