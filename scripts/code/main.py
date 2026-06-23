import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import sys
import os

from HMM import HMM_BaumWelch, HMM_Gibbs, strip_first_residue
from loaders import parse_fasta, load_empirical_params

def main():
    """
    Train both Baum-Welch and Gibbs HMM on real protein sequences.
    """
    
    # local path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file = "downsized"
    data_dir = os.path.join(repo_root, "data", "train_test_data", "hobohm1_ole_formula", file)
    
    # Configuration
    fasta_path = data_dir + "/train_dataset_fused_&_downsized_80.0%.txt"

    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    n_states = 4  # S, I, M, O
    max_iter = 200
    epsilon = 1e-5
    seed = 42
    empirical = True
    
    print()
    print("HMM Training Pipeline")
    print()
    
    # Load sequences and labels
    print(f"\nLoading sequences from {fasta_path}...")
    sequences, labels, headers = parse_fasta(fasta_path, alphabet=alphabet)
    sequences, labels, headers = strip_first_residue(sequences, labels, headers)

    if len(sequences) == 0:
        print("ERROR: No sequences loaded. Check file path and format.")
        sys.exit(1)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Total amino acids: {sum(len(s) for s in sequences)}")

    if empirical:
        #prepare empirical parameters
        #paths
        transition_path = data_dir + "/empirical_transition_prob_downsized"
        emission_path = data_dir + "/empirical_emission_prob_downsized"
        init_path = data_dir + "/empirical_init_prob_downsized"

        pi, A, B = load_empirical_params(transition_path, emission_path, init_path)



    
    # Train Baum-Welch
    print(f"\nTraining Baum-Welch HMM with {n_states} states...")
    bw = HMM_BaumWelch(number_of_states=n_states, alphabet=alphabet, seed=seed)

    if empirical: #overwrite
        bw.A = A
        bw.B = B
        bw.pi = pi

    bw_lls = bw.baum_welch(sequences, max_iter=max_iter, epsilon=epsilon)
    
    print(f"Iterations: {len(bw_lls)}")
    print(f"Final log-likelihood: {bw_lls[-1]:.4f}")
    
    # Train Gibbs
    print(f"\nTraining Gibbs HMM with {n_states} states...")
    gibbs = HMM_Gibbs(number_of_states=n_states, alphabet=alphabet, seed=seed)

    if empirical: #overwrite
        gibbs.A = A
        gibbs.B = B
        gibbs.pi = pi

    gibbs_lls = gibbs.gibbs(sequences, max_iter=max_iter, epsilon=epsilon)
    
    print(f"Iterations: {len(gibbs_lls)}")
    print(f"Final log-likelihood: {gibbs_lls[-1]:.4f}")
    
    # Save both models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    bw_model    = f"hmm_baum_welch_{timestamp}_{file}_{max_iter}_E.txt"
    gibbs_model = f"hmm_gibbs_{timestamp}_{file}_{max_iter}_E.txt"
    
    print(f"\nSaving models...")
    bw.output_model(bw_model)
    print(f"Baum-Welch: ../../results/models/{bw_model}")
    
    gibbs.output_model(gibbs_model)
    print(f"Gibbs: ../../results/models/{gibbs_model}")

  
    # Convergence figure (for poster)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    axes[0].plot(range(1, len(bw_lls)), bw_lls[1:], color="#4C72B0", linewidth=1.5)
    axes[0].set_title("Baum-Welch (exact EM)", fontsize=13)
    axes[0].set_xlabel("Iteration", fontsize=11)
    axes[0].set_ylabel("Training log-likelihood", fontsize=11)
    axes[0].annotate(
        f"Converged: {len(bw_lls)} iterations\nFinal LL: {bw_lls[-1]:.1f}",
        xy=(len(bw_lls), bw_lls[-1]),
        xytext=(0.45, 0.15), textcoords="axes fraction",
        fontsize=9, color="#4C72B0",
    )

    axes[1].plot(range(1, len(gibbs_lls)), gibbs_lls[1:], color="#C44E52", linewidth=1.5)
    axes[1].set_title("Gibbs sampling (stochastic EM)", fontsize=13)
    axes[1].set_xlabel("Iteration", fontsize=11)
    axes[1].set_ylabel("Training log-likelihood", fontsize=11)
    axes[1].annotate(
        f"Converged: {len(gibbs_lls)} iterations\nFinal LL: {gibbs_lls[-1]:.1f}",
        xy=(len(gibbs_lls), gibbs_lls[-1]),
        xytext=(0.45, 0.15), textcoords="axes fraction",
        fontsize=9, color="#C44E52",
    )

    fig.suptitle(
        f"Convergence: Baum-Welch vs. Gibbs sampling  "
        f"({n_states} hidden states, {len(sequences)} sequences)",
        fontsize=13,
    )
    fig.tight_layout()

    plot_path = f"/home/lunsusa/dtu/algorithms/project/TMHMM/results/models/convergence_{timestamp}_{file}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot: {plot_path}")

    print(f"\nTraining and saving complete!")
    

if __name__ == "__main__":
    main()