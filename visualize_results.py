import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_visualizations(csv_path="benchmark_progress.csv", output_dir="plots"):
    """Generate insight charts from the benchmark results."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Set plot style manually since we're not using seaborn
    plt.rcParams['axes.facecolor'] = '#f9f9f9'
    plt.rcParams['grid.color'] = 'white'
    
    # --- 1. Running Accuracy Plot ---
    print("Generating Running Accuracy plot...")
    df['Bee_Cum_Acc'] = df['BeeConsensus Correct'].expanding().mean() * 100
    df['SC_Cum_Acc'] = df['Self-consistency Correct'].expanding().mean() * 100
    df['Single_Cum_Acc'] = df['Single LLM Correct'].expanding().mean() * 100

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Bee_Cum_Acc'], label='BeeConsensus (Swarm)', color='#FFD700', linewidth=2.5)
    plt.plot(df.index, df['SC_Cum_Acc'], label='Self-consistency (N=5)', color='#999999', linestyle='--')
    plt.plot(df.index, df['Single_Cum_Acc'], label='Single LLM (Baseline)', color='#4169E1', linestyle=':')
    
    plt.title('TruthfulQA: Running Accuracy Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Question Number', fontsize=12)
    plt.ylabel('Cumulative Accuracy (%)', fontsize=12)
    plt.legend(frameon=True, facecolor='white')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'running_accuracy.png'), dpi=300)
    plt.close()

    # --- 2. Latency Distribution (Boxplot) ---
    print("Generating Latency Distribution plot...")
    plt.figure(figsize=(10, 6))
    methods = ['Single LLM Latency', 'Self-consistency Latency', 'BeeConsensus Latency']
    labels = ['Single LLM', 'Self-Con', 'BeeConsensus']
    
    data_to_plot = [df[m].dropna() / 1000 for m in methods] # Convert to seconds
    
    bp = plt.boxplot(data_to_plot, patch_artist=True, labels=labels)
    colors = ['#4169E1', '#999999', '#FFD700']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Inference Latency Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Seconds per Question', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'latency_distribution.png'), dpi=300)
    plt.close()

    # --- 3. BeeConsensus Confidence Distribution ---
    print("Generating Confidence analysis...")
    plt.figure(figsize=(8, 6))
    
    correct_conf = df[df['BeeConsensus Correct'] == 1]['BeeConsensus Confidence'].dropna()
    incorrect_conf = df[df['BeeConsensus Correct'] == 0]['BeeConsensus Confidence'].dropna()
    
    plt.hist(correct_conf, bins=15, alpha=0.5, label='Correct Answers', color='#2ECC40', density=True)
    plt.hist(incorrect_conf, bins=15, alpha=0.5, label='Incorrect Answers', color='#FF4136', density=True)
    
    plt.title('BeeConsensus: Confidence Score Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score (Waggle Dance Signal)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=300)
    plt.close()

    print(f"\nSuccess! 3 visualizations saved to the '{output_dir}/' directory.")
    print("1. running_accuracy.png")
    print("2. latency_distribution.png")
    print("3. confidence_analysis.png")

if __name__ == "__main__":
    generate_visualizations()
