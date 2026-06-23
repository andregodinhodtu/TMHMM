import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from argparse import ArgumentParser


parser = ArgumentParser(description="Creating plots from the output of the Viterbi script", add_help=True)
parser.add_argument("-mat", action="store", dest="conf_mat", type=str, help="Confusion matrix file")
parser.add_argument("-aa", action="store", dest="aa_file", type=str, help="Per Amino Acid accuracy file")
parser.add_argument("-m", action="store", dest="model_name", choices=["Baum-Welch", "Gibbs_Sampling"], help="Model training method (BW or Gibbs)")
parser.add_argument("-o", action="store", dest="output_dir", type=str, default=".", help="Path to the output directory")

args = parser.parse_args()
conf_mat = args.conf_mat
aa_file = args.aa_file
model_name = args.model_name
output_dir = args.output_dir


def plot_evaluation_results(confmat_file, aa_file, model_name, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")


    # Plotting the confusion matrix
    with open(confmat_file, 'r') as f:
        lines = f.readlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Global Accuracy:"):
            global_acc_str = line.split(":")[1].strip()
        if line.startswith("True\\Pred"):
            start_idx = i
            break

    df_conf = pd.read_csv(confmat_file, sep='\t', skiprows=start_idx, index_col=0)
    df_conf_norm = df_conf.div(df_conf.sum(axis=1), axis=0) * 100
    df_conf_norm = df_conf_norm.fillna(0) # If there's a 0 on a line

    desired_order = ["O", "M", "I", "S"]
    actual_order = [state for state in desired_order if state in df_conf_norm.index]
    df_conf_norm = df_conf_norm.reindex(index=actual_order, columns=actual_order)


    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df_conf_norm, annot=True, fmt=".1f", cmap="Blues", 
                     cbar_kws={'label': 'Prediction percentage per state'},
                     vmin=0, vmax=100)

    plt.title(f"Normalised Confusion Matrix - {model_name}\nGlobal Accuracy: {global_acc_str}",
              pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Predicted State", fontsize=12)
    plt.ylabel("True State ", fontsize=12)

    conf_plot_path = os.path.join(output_dir, f"{model_name}_confusion_plot.png")
    plt.tight_layout()
    plt.savefig(conf_plot_path, dpi=300)
    plt.close()



    # Plotting the per-AA accuracies
    df_aa = pd.read_csv(aa_file, sep='\t')

    df_aa['Acc_Float'] = (df_aa['Correct_Predictions'] / df_aa['Total_Count']) * 100
    df_aa['Acc_Float'] = df_aa['Acc_Float'].fillna(0)
    df_aa_sorted = df_aa.sort_values(by="Acc_Float", ascending=False)

    plt.figure(figsize=(12, 6))
    bars = sns.barplot(x='Amino_Acid', y='Acc_Float', data=df_aa_sorted, palette="viridis")
    mean_acc = (df_aa['Correct_Predictions'].sum() / df_aa['Total_Count'].sum()) * 100
    plt.axhline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Global mean ({mean_acc:.1f}%)')

    plt.title(f"Accuracy per Amino Acid - {model_name}", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.legend()


    aa_plot_path = os.path.join(output_dir, f"{model_name}_aa_accuracy_plot.png")
    plt.tight_layout()
    plt.savefig(aa_plot_path, dpi=300)
    plt.close()

if __name__ == "__main__":


    plot_evaluation_results(
        confmat_file= conf_mat,
        aa_file= aa_file,
        model_name= model_name,
	output_dir = output_dir
    )

