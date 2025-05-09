import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d  # For smoothing

# Improved color palette (colorblind-friendly)
colors = ['#4E79A7', '#F28E2B', '#E15759']  # Blue, Orange, Red
line_styles = ['-', '--', '-.']  # Different line styles for clarity

# Get current directory
folder = os.path.dirname(os.path.abspath(__file__))

# File names (ensure these match exactly)
file_names = [
    "offloading_metrics_1.xlsx",
    "offloading_metrics2.xlsx", 
    "offloading_metrics3.xlsx"
]

labels = [
    "Delay Weight=1, Energy Weight=0",
    "Delay Weight=0.8, Energy Weight=0.2", 
    "Delay Weight=0, Energy Weight=1"
]

plt.figure(figsize=(12, 6))

for idx, (file_name, label) in enumerate(zip(file_names, labels)):
    file_path = os.path.join(folder, file_name)
    try:
        df = pd.read_excel(file_path)
        
        # Apply Gaussian smoothing (adjust sigma for more/less smoothing)
        smoothed_cost = gaussian_filter1d(df["Avg_Cost"], sigma=2)
        
        # Replace zeros with small value to avoid log(0)
        y_values = np.where(df["Avg_Cost"] == 0, 1e-6, df["Avg_Cost"])
        
        plt.plot(df["Episode"], smoothed_cost, 
                label=label, 
                color=colors[idx],
                linestyle=line_styles[idx],
                linewidth=2.5,
                alpha=0.9)
        
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        continue

# Enhanced styling
plt.title("Average Computation Cost vs. Episodes", 
         fontsize=16, pad=20, fontweight='bold')
plt.xlabel("Episode Number", fontsize=14)
plt.ylabel("Average Computation Cost (Log Scale)", fontsize=14)

# Use log scale if there are extreme value differences
plt.yscale('log')  # Comment out if not needed

# Custom grid and ticks
plt.grid(True, which='both', linestyle=':', alpha=0.4)
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
plt.gca().set_yticklabels([f"{x:.1f}" for x in plt.gca().get_yticks()])

# Legend with shadow and background
legend = plt.legend(fontsize=12, framealpha=1, 
                   shadow=True, facecolor='white')
legend.get_frame().set_edgecolor('lightgray')

plt.tight_layout()

# Save high-quality image
plt.savefig(os.path.join(folder, "cost_comparison_enhanced.png"), 
           dpi=300, 
           bbox_inches='tight',
           transparent=False)
plt.show()