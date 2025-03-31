import pandas as pd
import matplotlib.pyplot as plt
import os # To check if file exists

# Read the CSV file
def read_csv(file_name):
    if not os.path.exists(file_name):
        print(f"Loss history file not found: {file_name}")
        return None
    try:
        data = pd.read_csv(file_name)
        # Basic validation
        if 'Epoch' not in data.columns or 'Train Loss' not in data.columns or 'Test Loss' not in data.columns:
             print(f"CSV file {file_name} is missing required columns (Epoch, Train Loss, Test Loss).")
             return None
        return data
    except Exception as e:
        print(f"Failed to read or parse CSV {file_name}: {e}")
        return None

# Plot the loss vs epochs
def plot_loss_vs_epochs(data):
    if data is not None and not data.empty:
        # Use a matplotlib style suitable for papers
        try:
            plt.style.use('seaborn-v0_8-paper')
        except OSError:
            print("Seaborn paper style not found, using default.")
            plt.style.use('default')

        # Create a figure and axis with larger size for a poster
        fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted size slightly

        # Plot training loss
        ax.plot(data['Epoch'], data['Train Loss'], label='Training Loss', marker='o', linestyle='-', color='blue', linewidth=2) # Adjusted linewidth

        # Plot testing loss (Uncommented)
        ax.plot(data['Epoch'], data['Test Loss'], label='Testing Loss', marker='s', linestyle='--', color='red', linewidth=2) # Adjusted linewidth

        # Set title and labels with larger font sizes
        ax.set_title('Loss vs Epochs', fontsize=24) # Adjusted fontsize
        ax.set_xlabel('Epochs', fontsize=20) # Adjusted fontsize
        ax.set_ylabel('Loss', fontsize=20) # Adjusted fontsize

        # Log scale for y-axis can be helpful if loss drops rapidly
        ax.set_yscale('log')
        print("Using log scale for Loss axis.")

        # Adjust tick label sizes
        ax.tick_params(axis='x', labelsize=16) # Adjusted fontsize
        ax.tick_params(axis='y', labelsize=16) # Adjusted fontsize

        # Add legend with larger font size
        ax.legend(loc='upper right', framealpha=0.7, fontsize=16) # Adjusted fontsize

        # Add grid for better readability
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Show the plot
        plt.tight_layout()
        plt.savefig('plot_loss_vs_epochs.png', dpi=300)  # Save with good DPI
        print("Plot saved to plot_loss_vs_epochs.png")
        plt.show()
    else:
        print("No data available to plot.")


# Main function
def main():
    file_name = 'loss-vs-epochs.csv'
    data = read_csv(file_name)
    plot_loss_vs_epochs(data)

if __name__ == "__main__":
    main()
