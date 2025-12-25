import matplotlib.pyplot as plt
import re
import argparse

def plot_loss(log_file):
    train_losses = []
    val_losses = []
    epochs = []
    
    with open(log_file, 'r') as f:
        content = f.read()
        
    # Regex to find: Epoch X | Train Loss: Y.YYYY | Val Loss: Z.ZZZZ
    pattern = r"Epoch (\d+) \| Train Loss: ([\d\.]+) \| Val Loss: ([\d\.]+)"
    matches = re.findall(pattern, content)
    
    for m in matches:
        epochs.append(int(m[0]))
        train_losses.append(float(m[1]))
        val_losses.append(float(m[2]))
        
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss: {log_file}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_file = log_file.replace('.log', '_loss.png').replace('scripts/', 'plots/')
    if '/' not in out_file: out_file = 'plots/' + out_file
    
    plt.savefig(out_file)
    print(f"Saved {out_file}. Final Train: {train_losses[-1]}, Final Val: {val_losses[-1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to log file")
    args = parser.parse_args()
    plot_loss(args.log_file)

