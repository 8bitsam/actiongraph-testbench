import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
def read_csv(file_name):
    try:
        data = pd.read_csv(file_name)
        return data
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return None

# Plot the loss vs epochs
def plot_loss_vs_epochs(data):
    if data is not None:
        # Use a matplotlib style suitable for papers
        plt.style.use('seaborn-v0_8-paper')
        
        # Create a figure and axis with larger size for a poster
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot training loss
        ax.plot(data['Epoch'], data['Train Loss'], label='Training Loss', marker='o', linestyle='-', color='blue', linewidth=3)
        
        # Plot testing loss
        # ax.plot(data['Epoch'], data['Test Loss'], label='Testing Loss', marker='s', linestyle='--', color='red', linewidth=3)
        
        # Set title and labels with larger font sizes
        ax.set_title('Loss vs Epochs', fontsize=30)
        ax.set_xlabel('Epochs', fontsize=25)
        ax.set_ylabel('Loss', fontsize=25)
        
        # Adjust tick label sizes
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        
        # Add legend with larger font size
        ax.legend(loc='upper right', framealpha=0.7, fontsize=20)
        
        # Show the plot
        plt.tight_layout()
        plt.savefig('plot.png', dpi=600)  # Save with high DPI for printing
        plt.show()

# Main function
def main():
    file_name = 'loss-vs-epochs.csv'
    data = read_csv(file_name)
    plot_loss_vs_epochs(data)

if __name__ == "__main__":
    main()

