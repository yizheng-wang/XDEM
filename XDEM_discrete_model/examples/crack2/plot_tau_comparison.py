import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

# Set random seed for reproducibility
import random
import torch
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def plot_tau_comparison():
    """
    Create comprehensive comparison plots for different tau values
    """
    
    # FEM results - mapping from tau to FEM K2 values
    tau_to_fem = {
        100: 141.4,
        200: 282.8, 
        300: 424.2,
        400: 565.6,
        500: 707.0,
        600: 848.4,
        700: 989.8,
        800: 1131.2,
        900: 1272.6,
        1000: 1414.0
    }
    
    # Load results from CSV if available
    results_csv_path = '../../result/crack2/crack_XDEM/different_load/k2_results_different_tau.csv'
    
    if os.path.exists(results_csv_path):
        print(f"Loading results from: {results_csv_path}")
        results_df = pd.read_csv(results_csv_path)
        
        # Extract data
        tau_values = results_df['tau'].values
        k2_true_values = results_df['K2_true'].values
        k2_j_values = results_df['K2_j'].values
        k2_disp_values = results_df['K2_disp'].values
        error_j_values = results_df['error_j'].values
        error_disp_values = results_df['error_disp'].values
        last_epochs = results_df['last_epoch'].values
        
        print(f"Loaded data for {len(tau_values)} tau values: {tau_values}")
    else:
        print(f"Results file not found: {results_csv_path}")
        print("Using theoretical values for plotting...")
        
        # Use theoretical values if CSV not available
        tau_values = np.array([100, 200, 300, 400, 500, 600])
        a = 0.5
        b = 1.0
        
        k2_true_values = []
        for tau in tau_values:
            a_b = a/b
            normalized_Param = tau * np.sqrt(np.pi * a)
            K2_true_coefficient = (1-0.025*a_b**2+0.06*a_b**4) * np.sqrt(1/np.cos(np.pi*a_b/2))
            K2_true = normalized_Param * K2_true_coefficient
            k2_true_values.append(K2_true)
        
        k2_true_values = np.array(k2_true_values)
        k2_j_values = None
        k2_disp_values = None
        error_j_values = None
        error_disp_values = None
        last_epochs = None
    
    # Get FEM values for the tau values we have
    fem_values = [tau_to_fem.get(tau, None) for tau in tau_values]
    fem_values = [val for val in fem_values if val is not None]
    
    # Create output directory
    output_dir = '../../result/crack2/crack_XDEM/different_load/'
    os.makedirs(output_dir, exist_ok=True)
    
    # =============================================================================
    # Plot 1: K2 Values Comparison
    # =============================================================================
    print("Creating K2 values comparison plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot theoretical K2 values
    plt.plot(tau_values, k2_true_values, 'k-', linewidth=3, label='Reference K2', marker='o', markersize=8)
    
    # Plot FEM K2 values if available
    if len(fem_values) > 0:
        plt.plot(tau_values[:len(fem_values)], fem_values, 'r-', linewidth=3, label='FEM K2', marker='s', markersize=8)
    
    # Plot XDEM K2 values if available
    if k2_j_values is not None:
        plt.plot(tau_values, k2_j_values, 'b-', linewidth=2, label='XDEM', marker='^', markersize=6)
    
    
    # Add value annotations for Reference K2 values
    for i, (tau_val, k2_val) in enumerate(zip(tau_values, k2_true_values)):
        plt.annotate(f'{k2_val:.1f}', 
                    (tau_val, k2_val), 
                    textcoords="offset points", 
                    xytext=(-15, 8), 
                    ha='center', 
                    fontsize=15, 
                    color='black',
                    fontweight='bold')
    
    # Add value annotations for FEM K2 values
    if len(fem_values) > 0:
        for i, (tau_val, k2_val) in enumerate(zip(tau_values[:len(fem_values)], fem_values)):
            plt.annotate(f'{k2_val:.1f}', 
                        (tau_val, k2_val), 
                        textcoords="offset points", 
                        xytext=(20, -20), 
                        ha='center', 
                        fontsize=15, 
                        color='red',
                        fontweight='bold')
    
    # Add value annotations for XDEM K2 values with relative error
    if k2_j_values is not None:
        for i, (tau_val, k2_val, k2_true, k2_fem) in enumerate(zip(tau_values, k2_j_values, k2_true_values, fem_values)):
            rel_error = abs(k2_val - k2_fem) / k2_fem * 100
            plt.annotate(f'{k2_val:.1f}\n({rel_error:.1f}%)', 
                        (tau_val, k2_val), 
                        textcoords="offset points", 
                        xytext=(-5, -63), 
                        ha='center', 
                        fontsize=15, 
                        color='blue',
                        fontweight='bold')
    
    plt.xlabel('Shear Stress τ (MPa)', fontsize=20)
    plt.ylabel('K2 (Stress Intensity Factor)', fontsize=20)
    plt.title('K2 Prediction vs Reference Values for Different Shear Stresses', fontsize=20)
    plt.ylim(0, 1000)  # 设置y轴坐标从0到1000
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    
    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_plot_path = os.path.join(output_dir, 'k2_comparison_different_tau_detailed.pdf')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"K2 comparison plot saved to: {comparison_plot_path}")
    
    # =============================================================================
    # Plot 2: Error Analysis
    # =============================================================================
    if error_j_values is not None:
        print("Creating error analysis plot...")
        
        plt.figure(figsize=(12, 8))
        
        # Plot J-integral errors
        plt.plot(tau_values, error_j_values, 'b-', linewidth=2, label='J-integral Error', marker='o', markersize=6)
        
        # Plot displacement errors if available
        if error_disp_values is not None:
            plt.plot(tau_values, error_disp_values, 'g-', linewidth=2, label='Displacement Error', marker='s', markersize=6)
        
        # Add value annotations for errors
        for i, (tau_val, error_val) in enumerate(zip(tau_values, error_j_values)):
            plt.annotate(f'{error_val:.1f}%', 
                        (tau_val, error_val), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=15, 
                        color='blue',
                        fontweight='bold')
        
        if error_disp_values is not None:
            for i, (tau_val, error_val) in enumerate(zip(tau_values, error_disp_values)):
                plt.annotate(f'{error_val:.1f}%', 
                            (tau_val, error_val), 
                            textcoords="offset points", 
                            xytext=(0, -15), 
                            ha='center', 
                            fontsize=15, 
                            color='green',
                            fontweight='bold')
        
        plt.xlabel('Shear Stress τ (MPa)', fontsize=20)
        plt.ylabel('Relative Error (%)', fontsize=20)

        plt.title('Prediction Error vs Shear Stress', fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=20)
        
        # 设置坐标轴刻度字体大小
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        
        plt.tight_layout()
        
        # Save the error plot
        error_plot_path = os.path.join(output_dir, 'k2_error_analysis_different_tau.pdf')
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error analysis plot saved to: {error_plot_path}")
    
    # =============================================================================
    # Plot 3: Training Epochs Analysis
    # =============================================================================
    if last_epochs is not None:
        print("Creating training epochs analysis plot...")
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(tau_values, last_epochs, 'm-', linewidth=2, label='Training Epochs', marker='o', markersize=6)
        
        # Add value annotations for epochs
        for i, (tau_val, epoch_val) in enumerate(zip(tau_values, last_epochs)):
            plt.annotate(f'{epoch_val}', 
                        (tau_val, epoch_val), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=15, 
                        color='purple',
                        fontweight='bold')
        
        plt.xlabel('Shear Stress τ (MPa)', fontsize=20)
        plt.ylabel('Training Epochs', fontsize=20)
        plt.title('Training Epochs vs Shear Stress', fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=20)
        
        # 设置坐标轴刻度字体大小
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        
        plt.tight_layout()
        
        # Save the epochs plot
        epochs_plot_path = os.path.join(output_dir, 'training_epochs_different_tau.pdf')
        plt.savefig(epochs_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training epochs plot saved to: {epochs_plot_path}")
    
    # =============================================================================
    # Plot 4: Combined Summary Plot
    # =============================================================================
    print("Creating combined summary plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: K2 Values
    ax1.plot(tau_values, k2_true_values, 'k-', linewidth=3, label='Reference K2', marker='o', markersize=6)
    if len(fem_values) > 0:
        ax1.plot(tau_values[:len(fem_values)], fem_values, 'r-', linewidth=3, label='FEM K2', marker='s', markersize=6)
    if k2_j_values is not None:
        ax1.plot(tau_values, k2_j_values, 'b-', linewidth=2, label='XDEM (J-integral)', marker='^', markersize=5)
    ax1.set_xlabel('Shear Stress τ (MPa)', fontsize=16)
    ax1.set_ylabel('K2 (Stress Intensity Factor)', fontsize=16)
    ax1.set_title('K2 Values Comparison', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Subplot 2: Error Analysis
    if error_j_values is not None:
        ax2.plot(tau_values, error_j_values, 'b-', linewidth=2, label='J-integral Error', marker='o', markersize=5)
        if error_disp_values is not None:
            ax2.plot(tau_values, error_disp_values, 'g-', linewidth=2, label='Displacement Error', marker='s', markersize=5)
    ax2.set_xlabel('Shear Stress τ (MPa)', fontsize=16)
    ax2.set_ylabel('Relative Error (%)', fontsize=16)
    ax2.set_title('Prediction Error', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Subplot 3: Training Epochs
    if last_epochs is not None:
        ax3.plot(tau_values, last_epochs, 'm-', linewidth=2, label='Training Epochs', marker='o', markersize=5)
    ax3.set_xlabel('Shear Stress τ (MPa)', fontsize=16)
    ax3.set_ylabel('Training Epochs', fontsize=16)
    ax3.set_title('Training Epochs', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    # Subplot 4: Statistics Summary
    ax4.axis('off')
    if error_j_values is not None:
        avg_error = np.mean(error_j_values)
        std_error = np.std(error_j_values)
        min_error = np.min(error_j_values)
        max_error = np.max(error_j_values)
        
        stats_text = f"""
        Error Statistics:
        Average: {avg_error:.2f}%
        Std Dev: {std_error:.2f}%
        Minimum: {min_error:.2f}%
        Maximum: {max_error:.2f}%
        
        Best Performance: τ = {tau_values[np.argmin(error_j_values)]} MPa
        Worst Performance: τ = {tau_values[np.argmax(error_j_values)]} MPa
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.suptitle('Comprehensive Analysis: Different Shear Stresses', fontsize=20)
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_path = os.path.join(output_dir, 'comprehensive_analysis_different_tau.pdf')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined summary plot saved to: {combined_plot_path}")
    
    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    plot_tau_comparison()
