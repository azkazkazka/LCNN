import sys
import argparse
import numpy as np
from calculate_modules import *
from calculate_metrics import calculate_minDCF_EER_CLLR_actDCF
import a_dcf

def read_scores(file_path):
    """Read scores from the given file path and return a dictionary with file names as keys and scores as values."""
    scores = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                file_name, score = parts
                scores[file_name] = float(score)
    return scores

def read_protocols(file_path):
    """Read protocols from the given file path and return a dictionary with file names as keys and labels ('bonafide' or 'spoof') as values."""
    protocols = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_name, label = parts[1], parts[4]
                protocols[file_name] = label
    return protocols

def find_metrics(scores_file_path, protocols_file_path):
    """Find the EER, minDCF, CLLR, and actDCF for given scores and protocols."""
    scores = read_scores(scores_file_path)
    protocols = read_protocols(protocols_file_path)
    
    # Align and label the scores with binary labels for bonafide=1 and spoof=0
    data = [(scores[file_name], 1 if protocols.get(file_name) == "bonafide" else 0) for file_name in scores if file_name in protocols]
    
    # Extract scores and labels
    cm_scores = np.array([score for score, _ in data])
    cm_keys = np.array([label for _, label in data])
    
    # Calculate metrics using the provided function
    minDCF, eer, cllr, actDCF, accuracy, cmatrix  = calculate_minDCF_EER_CLLR_actDCF(cm_scores, cm_keys)
    
    return minDCF, eer, cllr, actDCF, accuracy, cmatrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set scores file path and protocols file path
    parser.add_argument('--scores', type=str, default='./log_eval_score.txt')
    parser.add_argument('--protocols', type=str, default='./protocol.txt')

    args = parser.parse_args()
    scores_file_path = args.scores
    protocols_file_path = args.protocols
    
    minDCF, eer, cllr, actDCF, accuracy, cmatrix = find_metrics(scores_file_path, protocols_file_path)
