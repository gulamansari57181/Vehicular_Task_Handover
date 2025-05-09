import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

# High-contrast color palette (red to green gradient)
colors = ['#d62728',  # Strong red (max delay)
          '#ff7f0e',  # Orange
          '#e377c2',  # Pink
          '#17becf',  # Teal
          '#2ca02c']  # Strong green (max energy)

line_styles = ['-', '-', '-', '-', (0, (3, 1, 1, 1))]  # Distinct line styles
line_widths = [2.5, 2.5, 2.0, 2.0, 2.5]  # Emphasize first and last

file_names = ["offloading_metrics_2.xlsx", "offloading_metrics_2.xlsx",
             "offloading_metrics_3.xlsx", "offloading_metrics_4.xlsx",
             "offloading_metrics_5.xlsx"]

labels = ["Delay=0.5, Energy=1.0", 
          "Delay=0.8, Energy=0.2",
          "Delay=1, Energy=0",
          "Delay=0.2, Energy=0.8",  
          "Delay=0.0, Energy=1.0"]

folder = r"D:/M Tech Assignments/Cost_Data/"

plt.figure(figsize=(12, 7))  # Slightly taller for legend space

max_values = []
min_values = []

for idx, (file_name, label) in enumerate(zip(file_names, labels)):
    file_path = os.path.join(folder, file_name)
    try:
        df = pd.read_excel(file_path)
        
        # Apply smoothing while preserving peaks
        smoothed_values = gaussian_filter1d(df["Avg_Cost"], sigma=1.2)
        
        # Track min/max for scaling
        max_values.append(np.max(smoothed_values))
        min_values.append(np.min(smoothed_values))
            
        plt.plot(df["Episode"], smoothed_values,
                label=label,
                color=colors[idx],
                linestyle=line_styles[idx],
                linewidth=line_widths[idx],
                alpha=0.9)
        
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        continue

# Visual confirmation of delay priority
if max_values[0] >= max(max_values[1:]):
    print("✓ Validation: Delay=1.0 shows highest costs")
else:
    print("⚠ Warning: Check delay-weight relationship")

# Enhanced styling
plt.title("Computation Cost Trade-off: Delay vs Energy Optimization", 
         fontsize=15, pad=18, fontweight='bold')
plt.xlabel("Training Episodes", fontsize=13)
plt.ylabel("Computation Cost", fontsize=13)

# Dynamic axis scaling
plt.ylim(min(min_values)*0.9, max(max_values)*1.15)  # 10-15% padding

# Professional grid and legend
plt.grid(True, linestyle='--', alpha=0.4)
legend = plt.legend(fontsize=11, framealpha=0.9, 
                   loc='upper center', 
                   bbox_to_anchor=(0.5, -0.15),
                   ncol=3)
legend.get_frame().set_edgecolor('black')

plt.tight_layout()

# Save high-quality output
output_path = os.path.join(folder, "cost_tradeoff_contrast.png")
plt.savefig(output_path, dpi=350, bbox_inches='tight')
print(f"High-contrast plot saved to:\n{output_path}")
plt.show()