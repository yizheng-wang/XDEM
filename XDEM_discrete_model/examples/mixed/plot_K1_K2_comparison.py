import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_K1_K2_comparison():
    """
    Program specifically for plotting K1 and K2 comparison charts
    Supports reading data from CSV files or using predefined theoretical values
    """
    
    # Set font for better display
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Theoretical values data (values you have filled in)
    K1_theory_values = {
        15: 136.3,
        30: 117.6,
        45: 92.45,
        60: 67.2,
        75: 49.23
    }
    
    K2_theory_values = {
        15: 17.26,
        30: 33.1,
        45: 40.15,
        60: 35.32,
        75: 20.30
    }
    
    # Angle list
    beta_angles = [15, 30, 45, 60, 75]
    
    # Read K1 and K2 historical data from subfolders for each angle
    base_path = '../../result/crack_mix/crack_XDEM/a_0.5'
    K1_calculated = []
    K2_calculated = []
    K1_errors = []
    K2_errors = []
    
    print("Reading K1 and K2 data from subfolders for each angle...")
    
    for angle in beta_angles:
        # Build historical data file path for each angle
        history_file = f'{base_path}/different_beta/beta_{angle}/crack_k1_history.csv'
        
        if os.path.exists(history_file):
            print(f"Reading data for angle {angle}°: {history_file}")
            try:
                # Read historical data
                history_df = pd.read_csv(history_file)
                
                # Use last iteration result (more objective)
                if len(history_df) > 0:
                    # Get the last epoch result
                    last_result = history_df.iloc[-1]
                    
                    # Calculate errors for the last epoch
                    K1_theory_val = K1_theory_values[angle]
                    K2_theory_val = K2_theory_values[angle]
                    
                    K1_error = np.abs(last_result['K_I_J'] - K1_theory_val) / K1_theory_val * 100
                    K2_error = np.abs(last_result['K_II_J'] - K2_theory_val) / K2_theory_val * 100
                    total_error = K1_error + K2_error
                    
                    K1_calculated.append(last_result['K_I_J'])
                    K2_calculated.append(last_result['K_II_J'])
                    K1_errors.append(K1_error)
                    K2_errors.append(K2_error)
                    
                    print(f"  Angle {angle}°: K1={last_result['K_I_J']:.2f}, K2={last_result['K_II_J']:.2f}, Total Error={total_error:.2f}% (Last Epoch: {last_result['epoch']})")
                else:
                    print(f"  Angle {angle}°: Historical data is empty")
                    K1_calculated.append(0.0)
                    K2_calculated.append(0.0)
                    K1_errors.append(0.0)
                    K2_errors.append(0.0)
                    
            except Exception as e:
                print(f"  Angle {angle}°: Error reading data - {e}")
                K1_calculated.append(0.0)
                K2_calculated.append(0.0)
                K1_errors.append(0.0)
                K2_errors.append(0.0)
        else:
            print(f"  Angle {angle}°: File does not exist - {history_file}")
            K1_calculated.append(0.0)
            K2_calculated.append(0.0)
            K1_errors.append(0.0)
            K2_errors.append(0.0)
    
    # Convert to numpy arrays
    K1_calculated = np.array(K1_calculated)
    K2_calculated = np.array(K2_calculated)
    K1_errors = np.array(K1_errors)
    K2_errors = np.array(K2_errors)
    
    print(f"\nSuccessfully read data for {len([x for x in K1_calculated if x != 0])} angles")
    
    # Extract theoretical values
    K1_theory = np.array([K1_theory_values[angle] for angle in beta_angles])
    K2_theory = np.array([K2_theory_values[angle] for angle in beta_angles])
    

    # Create separate K1 and K2 comparison chart
    plt.figure(figsize=(12, 8))
    
    # Plot four curves
    plt.plot(beta_angles, K1_theory, 'o-', label='K1 Reference', linewidth=3, markersize=8, color='blue')
    plt.plot(beta_angles, K1_calculated, 's--', label='K1 XDEM Prediction', linewidth=3, markersize=8, color='red')
    plt.plot(beta_angles, K2_theory, '^-', label='K2 Reference', linewidth=3, markersize=8, color='green')
    plt.plot(beta_angles, K2_calculated, 'v--', label='K2 XDEM Prediction', linewidth=3, markersize=8, color='orange')
    
    plt.xlabel('Beta Angle (degrees)', fontsize=14)
    plt.ylabel('Stress Intensity Factor (MPa√mm)', fontsize=14)
    plt.title('K1 and K2: Reference vs XDEM Prediction Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.xlim(10, 80)
    plt.ylim(0, 150)
    plt.xticks(beta_angles)
    
    # Add value annotations
    for i, angle in enumerate(beta_angles):
        plt.annotate(f'{K1_theory[i]:.1f}', (angle, K1_theory[i]), 
                    textcoords="offset points", xytext=(0, 20), ha='center', fontsize=20, color='blue', fontweight='bold')
        plt.annotate(f'{K1_calculated[i]:.1f}', (angle, K1_calculated[i]), 
                    textcoords="offset points", xytext=(0, -30), ha='center', fontsize=20, color='red', fontweight='bold')
        plt.annotate(f'{K2_theory[i]:.1f}', (angle, K2_theory[i]), 
                    textcoords="offset points", xytext=(0, 20), ha='center', fontsize=20, color='green', fontweight='bold')
        plt.annotate(f'{K2_calculated[i]:.1f}', (angle, K2_calculated[i]), 
                    textcoords="offset points", xytext=(0, -20), ha='center', fontsize=20, color='orange', fontweight='bold')
    
    plt.tight_layout()
    
    # Save separate comparison chart
    single_plot_path = '../../result/crack_mix/crack_XDEM/a_0.5/different_beta/K1_K2_single_comparison.pdf'
    plt.savefig(single_plot_path, dpi=300, bbox_inches='tight')




if __name__ == "__main__":
    print("="*60)
    print("K1 and K2 Stress Intensity Factor Comparison Analysis Program")
    print("="*60)
    
    # Run main analysis
    results = plot_K1_K2_comparison()
    
    print("\n" + "="*60)
    print("Plotting training history curves...")
    print("="*60)
    
    # Plot training history
    
    print("\nProgram execution completed!")
    print("Generated chart files:")
    print("1. K1_K2_comprehensive_analysis.png - Four-subplot comprehensive analysis")
    print("2. K1_K2_single_comparison.pdf - Single chart four-curve comparison")
    print("3. K1_K2_training_history.png - Training history curves for each angle")
    print("4. K1_K2_summary_table.csv - Results summary table")
