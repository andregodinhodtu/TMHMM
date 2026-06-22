import numpy as np
from HMM import HMM_BaumWelch, HMM_Gibbs, parse_fasta
from datetime import datetime
import sys

def main():
    """
    Train both Baum-Welch and Gibbs HMM on real protein sequences.
    """
    
    # Configuration
    fasta_path = "/home/lunsusa/dtu/algorithms/project/TMHMM/data/train_test_data/hobohm1_ole_formula/default/train_dataset_fused_80.0%.txt"
    file = "fused"
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    n_states = 4  # S, I, M, O
    max_iter = 500
    epsilon = 1e-5
    seed = 42
    
    print()
    print("HMM Training Pipeline")
    print()
    
    # Load sequences and labels
    print(f"\nLoading sequences from {fasta_path}...")
    sequences, labels, headers = parse_fasta(fasta_path, alphabet=alphabet)
    
    if len(sequences) == 0:
        print("ERROR: No sequences loaded. Check file path and format.")
        sys.exit(1)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Total amino acids: {sum(len(s) for s in sequences)}")
    
    # Train Baum-Welch
    print(f"\nTraining Baum-Welch HMM with {n_states} states...")
    bw = HMM_BaumWelch(number_of_states=n_states, alphabet=alphabet, seed=seed)
    bw_lls = bw.baum_welch(sequences, max_iter=max_iter, epsilon=epsilon)
    
    print(f"Iterations: {len(bw_lls)}")
    print(f"Final log-likelihood: {bw_lls[-1]:.4f}")
    
    # Train Gibbs
    print(f"\nTraining Gibbs HMM with {n_states} states...")
    gibbs = HMM_Gibbs(number_of_states=n_states, alphabet=alphabet, seed=seed)
    gibbs_lls = gibbs.gibbs(sequences, max_iter=max_iter, epsilon=epsilon)
    
    print(f"Iterations: {len(gibbs_lls)}")
    print(f"Final log-likelihood: {gibbs_lls[-1]:.4f}")
    
    # Save both models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    bw_model = f"hmm_baum_welch_{timestamp}_{file}_{max_iter}.txt"
    gibbs_model = f"hmm_gibbs_{timestamp}_{file}_{max_iter}.txt"
    
    print(f"\nSaving models...")
    bw.output_model(bw_model)
    print(f"Baum-Welch: ../../results/models/{bw_model}")
    
    gibbs.output_model(gibbs_model)
    print(f"Gibbs: ../../results/models/{gibbs_model}")
    
    print(f"\nTraining and saving complete!")
    

if __name__ == "__main__":
    main()