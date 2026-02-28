import sys
import os
sys.path.append((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

def quick_plot():
    """Quick plot of K1 comparison results"""
    
    # Load XDEM results
    xdem_results_path = '../../result/crack1/crack_XDEM/different_a/k1_results_different_a.csv'
    dem_results_path = '../../result/crack1/crack_DEM/different_a/k1_results_different_a.csv'
    
    if not os.path.exists(xdem_results_path):
        print(f"Error: XDEM results file not found at {xdem_results_path}")
        return
    
    if not os.path.exists(dem_results_path):
        print(f"Error: DEM results file not found at {dem_results_path}")
        return
    
    # Load both datasets
    df_xdem = pd.read_csv(xdem_results_path)
    df_dem = pd.read_csv(dem_results_path)
    print(f"Loaded XDEM results for {len(df_xdem)} crack lengths: {df_xdem['a'].tolist()}")
    print(f"Loaded DEM results for {len(df_dem)} crack lengths: {df_dem['a'].tolist()}")
    
    # Create the main comparison plot
    plt.figure(figsize=(12, 8))
    
    # Extract XDEM data
    a_values = df_xdem['a'].values
    k1_true_values = df_xdem['K1_true'].values
    k1_xdem_values = df_xdem['K1_j'].values
    
    # Extract DEM data (assuming same crack lengths)
    k1_dem_values = df_dem['K1_j'].values
    
    # Plot K1 values comparison
    plt.plot(a_values, k1_true_values, 'k-', linewidth=3, label='True K1', marker='o', markersize=8)
    plt.scatter(a_values, k1_xdem_values, c='r', marker='s', s=100, label='XDEM')
    plt.scatter(a_values, k1_dem_values, c='b', marker='^', s=100, label='DEM')
    
    # Add value annotations for True K1 values
    for i, (a_val, k1_val) in enumerate(zip(a_values, k1_true_values)):
        plt.annotate(f'{k1_val:.1f}', 
                    (a_val, k1_val), 
                    textcoords="offset points", 
                    xytext=(-13, 17), 
                    ha='center', 
                    fontsize=15, 
                    color='black',
                    fontweight='bold')
    
    # Add value annotations for XDEM K1 values with relative error
    for i, (a_val, k1_val, k1_true) in enumerate(zip(a_values, k1_xdem_values, k1_true_values)):
        rel_error = abs(k1_val - k1_true) / k1_true * 100
        # 最后一个值需要更往下面
        if i == len(a_values) - 1:
            xytext_offset = (10, -65)
        else:
            xytext_offset = (10, -45)
        plt.annotate(f'{k1_val:.1f}\n({rel_error:.1f}%)', 
                    (a_val, k1_val), 
                    textcoords="offset points", 
                    xytext=xytext_offset, 
                    ha='center', 
                    fontsize=15, 
                    color='red',
                    fontweight='bold')
    
    # Add value annotations for DEM K1 values with relative error
    for i, (a_val, k1_val, k1_true) in enumerate(zip(a_values, k1_dem_values, k1_true_values)):
        rel_error = abs(k1_val - k1_true) / k1_true * 100
        if i == len(a_values) - 1:
            xytext_offset = (-3, 15)
        else:
            xytext_offset = (-3, 45)
        plt.annotate(f'{k1_val:.1f}\n({rel_error:.1f}%)', 
                    (a_val, k1_val), 
                    textcoords="offset points", 
                    xytext=xytext_offset, 
                    ha='center', 
                    fontsize=15, 
                    color='blue',
                    fontweight='bold')
    
    plt.xlabel('Crack Length (a)', fontsize=20)
    plt.ylabel('K1 (Stress Intensity Factor)', fontsize=20)
    plt.title('K1 Prediction vs True Values for Different Crack Lengths', fontsize=20)
    plt.ylim(bottom=0, top=600)  # 设置y坐标最小值为0
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    
    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = '../../result/crack1/k1_comparison_quick.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Quick comparison plot saved to: {plot_path}")
    
    # Print quick statistics for both methods
    error_xdem = df_xdem['error_j'].values
    error_dem = df_dem['error_j'].values
    
    print(f"\nQuick Statistics:")
    print(f"XDEM J-integral error: {np.mean(error_xdem):.2f}% ± {np.std(error_xdem):.2f}%")
    print(f"DEM J-integral error: {np.mean(error_dem):.2f}% ± {np.std(error_dem):.2f}%")
    print(f"Best XDEM: {np.min(error_xdem):.2f}% (a={df_xdem.loc[np.argmin(error_xdem), 'a']:.1f})")
    print(f"Best DEM: {np.min(error_dem):.2f}% (a={df_dem.loc[np.argmin(error_dem), 'a']:.1f})")

if __name__ == "__main__":
    quick_plot()
