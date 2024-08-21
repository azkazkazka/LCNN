import re
import matplotlib.pyplot as plt

def parse_logs(file_path):
    fold_data = {}
    current_fold = None
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("FOLD"):
                current_fold = int(re.search(r'\d+', line).group())
                fold_data[current_fold] = {'epoch': [], 'train_loss': [], 'dev_loss': []}
            elif re.match(r'\s*\d+\s*\|\s*\d+\.\d+\s*\|\s*\d+\.\d+\s*\|\s*\d+\.\d+\s*\|', line):
                parts = line.split('|')
                epoch = int(parts[0].strip())
                train_loss = float(parts[2].strip())
                dev_loss = float(parts[3].strip())
                fold_data[current_fold]['epoch'].append(epoch)
                fold_data[current_fold]['train_loss'].append(train_loss)
                fold_data[current_fold]['dev_loss'].append(dev_loss)
                
    return fold_data

def plot_loss(fold_data, fold_on=True):
    plt.figure(figsize=(12, 8))
    for fold, data in fold_data.items():
        prefix = ""
        if fold_on:
            prefix = f"Fold {fold} "
        plt.plot(data['epoch'], data['train_loss'], label=f'{prefix}Train Loss')
        plt.plot(data['epoch'], data['dev_loss'], label=f'{prefix}Dev Loss', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Development Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./plot_loss_test.jpg")
    plt.show()

def plot_loss_per_fold(fold_data, save_folder):
    for fold, data in fold_data.items():
        plt.figure(figsize=(12, 8))
        
        plt.plot(data['epoch'], data['train_loss'], label='Train Loss')
        plt.plot(data['epoch'], data['dev_loss'], label='Dev Loss', linestyle='--')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss per Epoch for Fold {fold}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_folder}/fold_{fold}_loss_plot.png")
        plt.show()


# File path to your log file
log_file_path = './loss_test.txt'

# Parse the logs
fold_data = parse_logs(log_file_path)

# Plot the loss graphs
plot_loss(fold_data, fold_on=False)


# # Folder to save the plots
# save_folder = '.'

# # Plot the loss graphs for each fold and save the plots
# plot_loss_per_fold(fold_data, save_folder)