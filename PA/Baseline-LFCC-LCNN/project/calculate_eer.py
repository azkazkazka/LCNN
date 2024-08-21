import sys
import argparse
from sklearn.metrics import roc_curve
import numpy as np

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

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer

def calculate_accuracy(y_true, y_scores):
    y_pred = [np.round(score) for score in y_scores]
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    return accuracy

def find_eer_accuracy(scores_file_path, protocols_file_path):
    """Find the EER for given scores and protocols."""
    scores = read_scores(scores_file_path)
    protocols = read_protocols(protocols_file_path)
    
    # Align and label the scores with binary labels for bonafide=1 and spoof=0
    data = [(scores[file_name], 1 if protocols.get(file_name) == "bonafide" else 0) for file_name in scores if file_name in protocols]
    
    # Sort the data based on scores to simplify thresholding
    data.sort(key=lambda x: x[0])
    y_scores = [score for score, _ in data]
    y_true = [label for _, label in data]
    
    eer = calculate_eer(y_true, y_scores)

    accuracy = calculate_accuracy(y_true, y_scores)
    
    return eer, accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set scores file path and protocols file path
    parser.add_argument('--scores', type=str, default='/home/sarah.azka/speech/LFCC/PA/Baseline-LFCC-LCNN/project/baseline_PA/log_eval_score.txt')
    parser.add_argument('--protocols', type=str, default='/home/sarah.azka/speech/NEW_DATA_TA_PA/protocol.txt')

    args = parser.parse_args()
    scores_file_path = args.scores
    protocols_file_path = args.protocols
    
    # Example usage (replace with the actual paths to your files)
    eer, accuracy = find_eer_accuracy(scores_file_path, protocols_file_path)
    print(f"The EER is: {eer}")
    print(f"The accuracy is: {accuracy}")
