# import sys
# import argparse

# def read_scores(file_path):
#     """Read scores from the given file path and return a dictionary with file names as keys and scores as values."""
#     scores = {}
#     with open(file_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 file_name, score = parts
#                 scores[file_name] = float(score)
#     return scores

# def read_protocols(file_path):
#     """Read protocols from the given file path and return a dictionary with file names as keys and labels ('bonafide' or 'spoof') as values."""
#     protocols = {}
#     with open(file_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split()
#             if len(parts) >= 5:
#                 file_name, label = parts[1], parts[4]
#                 protocols[file_name] = label
#     return protocols

# def calculate_far_frr(data, threshold):
#     """Calculate FAR and FRR given a threshold."""
#     false_accepts = sum(1 for score, label in data if score >= threshold and label == 0)
#     false_rejects = sum(1 for score, label in data if score < threshold and label == 1)
#     num_bonafide = sum(1 for _, label in data if label == 1)
#     num_spoof = len(data) - num_bonafide
#     far = false_accepts / num_spoof if num_spoof else 0
#     frr = false_rejects / num_bonafide if num_bonafide else 0
#     return far, frr

# def find_eer(scores_file_path, protocols_file_path):
#     """Find the EER for given scores and protocols."""
#     scores = read_scores(scores_file_path)
#     protocols = read_protocols(protocols_file_path)
    
#     # Align and label the scores with binary labels for bonafide=1 and spoof=0
#     data = [(scores[file_name], 1 if protocols.get(file_name) == "bonafide" else 0) for file_name in scores if file_name in protocols]
    
#     # Sort the data based on scores to simplify thresholding
#     data.sort(key=lambda x: x[0])
    
#     thresholds = [score for score, _ in data]
#     eer = None
#     best_threshold = None
#     min_difference = float('inf')
#     for threshold in thresholds:
#         far, frr = calculate_far_frr(data, threshold)
#         if abs(far - frr) < min_difference:
#             min_difference = abs(far - frr)
#             eer = (far + frr) / 2  # EER is the point where FAR and FRR are equal
#             best_threshold = threshold
    
#     return eer, best_threshold

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # set scores file path and protocols file path
#     parser.add_argument('--scores', type=str, default='/home/sarah.azka/speech/LFCC/LA/Baseline-LFCC-LCNN/project/baseline_LA/log_eval_new_score.txt')
#     parser.add_argument('--protocols', type=str, default='/home/sarah.azka/speech/NEW_DATA_LA/protocol.txt')

#     args = parser.parse_args()
#     scores_file_path = args.scores
#     protocols_file_path = args.protocols
    
#     # Example usage (replace with the actual paths to your files)
#     eer, best_threshold = find_eer(scores_file_path, protocols_file_path)
#     print(f"The EER is: {eer}")
#     print(f"With threshold of: {best_threshold}")


import sys
import argparse
import numpy as np
from calculate_modules import *
from test_evaluation_metrics import calculate_minDCF_EER_CLLR_actDCF
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
    parser.add_argument('--scores', type=str, default='/home/sarah.azka/speech/LFCC/LA/Baseline-LFCC-LCNN/project/baseline_LA/log_eval_cv_only_to_prosa_score.txt')
    parser.add_argument('--protocols', type=str, default='/home/sarah.azka/speech/NEW_DATA_LA/protocol.txt')

    args = parser.parse_args()
    scores_file_path = args.scores
    protocols_file_path = args.protocols
    
    # Example usage (replace with the actual paths to your files)
    minDCF, eer, cllr, actDCF, accuracy, cmatrix = find_metrics(scores_file_path, protocols_file_path)
    print(f"minDCF: {minDCF}")
    print(f"EER: {eer}")
    print(f"CLLR: {cllr}")
    print(f"actDCF: {actDCF}")
    print(f"accuracy: {accuracy}")
    print(f"confusion matrix: {cmatrix}")

    print(f"from {scores_file_path} and {protocols_file_path}")
