

import numpy as np
from datetime import datetime


class HMM_Gibbs():
    """
    Hidden Markov Model trained via Gibbs sampling.

    At each iteration, for each sequence a state path is sampled from the
    posterior given the current parameters (forward-filtering /
    backward-sampling). Parameters are then re-estimated by counting over
    all sampled paths.

    Attributes
    ----------
    n_states : int
    alphabet : str
    n_symbols : int
    pi : np.ndarray (n_states,)
    A  : np.ndarray (n_states, n_states)
    B  : np.ndarray (n_states, n_symbols)
    log_likelihoods : list of float
        Populated after calling gibbs().
    """

    def __init__(self, number_of_states, alphabet, seed=42):

        np.random.seed(seed)

        self.n_states  = number_of_states
        self.alphabet  = alphabet
        self.n_symbols = len(set(alphabet))

        # Random initialisation like HMM
        self.pi = np.random.dirichlet(np.ones(self.n_states))
        self.A  = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        self.B  = np.random.dirichlet(np.ones(self.n_symbols), size=self.n_states)

        self.log_likelihoods = []

    # many things copied from HMM_training.py

    def encode(self, sequence):
        """Map amino acid string to list of ints (alphabet order, not sorted)."""
        aa_to_idx = {aa: i for i, aa in enumerate(self.alphabet)}
        return [aa_to_idx[aa] for aa in sequence]

    # Forward pass (scaled) for path sampling and log-likelihood
    def _forward(self, input_encode):

        T = len(input_encode)
        alpha   = np.zeros((self.n_states, T))
        scaling = np.zeros(T)

        alpha[:, 0]  = self.pi * self.B[:, input_encode[0]]
        scaling[0]   = alpha[:, 0].sum()
        alpha[:, 0] /= scaling[0]

        for t in range(1, T):
            alpha[:, t]  = self.B[:, input_encode[t]] * (self.A.T @ alpha[:, t-1])
            scaling[t]   = alpha[:, t].sum()
            alpha[:, t] /= scaling[t]

        return alpha, scaling

    # Sample a state path given current parameters
    def _sample_path(self, input_encode):
        """
        Forward pass to get alpha, then sample a path backwards.

        At each t, sample z_t from:
            P(z_t | z_{t+1}, obs) ~ alpha[z_t, t] * A[z_t, z_{t+1}]
        """

        T = len(input_encode)
        alpha, _ = self._forward(input_encode)
        path = np.zeros(T, dtype=int)

        # Sample last state from alpha[:, T-1]
        probs = alpha[:, T-1]
        probs = probs / probs.sum()
        path[T-1] = np.random.choice(self.n_states, p=probs)

        # Sample backwards
        for t in range(T-2, -1, -1):
            probs = alpha[:, t] * self.A[:, path[t+1]]
            probs = probs / probs.sum()
            path[t] = np.random.choice(self.n_states, p=probs)

        return path

    # Re-estimate parameters from sampled paths
    def _update_parameters(self, all_paths, all_encoded):

        # pi: fraction of sequences starting in each state
        pi_counts = np.zeros(self.n_states)
        for path in all_paths:
            pi_counts[path[0]] += 1
        # tiny pseudocount avoids a state permanently dying (0 probability
        # mass -> can never be sampled into again)
        self.pi = (pi_counts + 1e-3) / (pi_counts.sum() + self.n_states * 1e-3)

        # A: transition counts
        A_counts = np.zeros((self.n_states, self.n_states))
        for path in all_paths:
            for t in range(len(path) - 1):
                A_counts[path[t], path[t+1]] += 1
        A_counts += 1e-3
        self.A = A_counts / A_counts.sum(axis=1, keepdims=True)

        # B: emission counts
        B_counts = np.zeros((self.n_states, self.n_symbols))
        for path, enc in zip(all_paths, all_encoded):
            for t, obs in enumerate(enc):
                B_counts[path[t], obs] += 1
        B_counts += 1e-3
        self.B = B_counts / B_counts.sum(axis=1, keepdims=True)

    # Log-likelihood (for convergence tracking and comparison with Baum-Welch)
    def _log_likelihood(self, all_encoded):
        ll = 0.0
        for enc in all_encoded:
            _, scaling = self._forward(enc)
            ll += np.sum(np.log(scaling))
        return ll

    # Main training loop
    def gibbs(self, sequences, max_iter=100, epsilon=1e-4, verbose=True):
        """
        Train the HMM via Gibbs sampling.

        Parameters
        ----------
        sequences : list of list of int
            Encoded sequences (same format as Baum-Welch / HMM_efficient).
        max_iter : int
            Maximum number of iterations.
        epsilon : float
            Convergence threshold on log-likelihood (checked on a running
            average to smooth out sampling noise).

        Returns
        -------
        log_likelihoods : list of float
        """

        self.log_likelihoods = []

        for iteration in range(max_iter):

            # Sample a path for every sequence
            all_paths = [self._sample_path(enc) for enc in sequences]

            # Re-estimate parameters from sampled paths
            self._update_parameters(all_paths, sequences)

            # Track log-likelihood
            ll = self._log_likelihood(sequences)
            self.log_likelihoods.append(ll)
            if verbose:
                print(f"Iteration {iteration + 1} | log-likelihood: {ll:.4f}")

            # Convergence check on a short moving average, since a single
            # sampled path makes the raw log-likelihood noisy
            if iteration >= 10:
                prev_avg = np.mean(self.log_likelihoods[-10:-5])
                curr_avg = np.mean(self.log_likelihoods[-5:])
                if abs(curr_avg - prev_avg) < epsilon:
                    if verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break

        return self.log_likelihoods

    # Evaluate copied
    def evaluate(self, encoded_seq, true_labels):
        """
        Evaluate using posterior decoding (gamma).
        """

        T = len(encoded_seq)
        alpha, scaling = self._forward(encoded_seq)

        # Backward pass for gamma
        beta = np.zeros((self.n_states, T))
        beta[:, -1] = 1.0
        for t in range(T-2, -1, -1):
            beta[:, t] = self.A @ (self.B[:, encoded_seq[t+1]] * beta[:, t+1])
            s = beta[:, t].sum()
            if s > 0:
                beta[:, t] /= s

        gamma = alpha * beta
        gamma /= gamma.sum(axis=0, keepdims=True)

        true_labels = np.array(true_labels)

        ce = -np.mean(np.log(gamma[true_labels, np.arange(T)] + 1e-10))
        predicted = np.argmax(gamma, axis=0)
        accuracy = np.mean(predicted == true_labels)

        return ce, accuracy

    # Save / load has the same file format as HMM_efficient.output_model() :)

    def output_model(self, output_filename):
        output_default_path = "../../results/models/" + output_filename

        with open(output_default_path, "w") as f:
            f.write(f"> Hidden Markov Model: {output_filename}\n")
            f.write(f"> Training method: Gibbs sampling\n")
            f.write(f"> Alphabet: {self.alphabet}\n")
            f.write(f"> Number of hidden states: {self.n_states}\n")
            f.write(f"> Number of symbols: {self.n_symbols}\n\n")

            f.write("> Initial state probabilities (pi)\n")
            f.write(f"> Shape: {self.pi.shape}\n")
            for i, val in enumerate(self.pi):
                f.write(f"state_{i}\t{val:.6f}\n")
            f.write("\n")

            f.write("> Transition matrix (A)\n")
            f.write(f"> Shape: {self.A.shape}\n")
            for i in range(self.n_states):
                row = "\t".join(f"{val:.6f}" for val in self.A[i])
                f.write(f"state_{i}\t{row}\n")
            f.write("\n")

            f.write("> Emission matrix (B)\n")
            f.write(f"> Shape: {self.B.shape}\n")
            for i in range(self.n_states):
                row = "\t".join(f"{val:.6f}" for val in self.B[i])
                f.write(f"state_{i}\t{row}\n")

    @classmethod
    def load_model(cls, input_filename):
        input_default_path = "../../results/models/" + input_filename
        with open(input_default_path, "r") as f:
            lines = f.readlines()

        alphabet, n_states = None, None
        for line in lines:
            if line.startswith("> Alphabet:"):
                alphabet = line.split(": ")[1].strip()
            if line.startswith("> Number of hidden states:"):
                n_states = int(line.split(": ")[1].strip())
            if alphabet and n_states:
                break

        hmm = cls(n_states, alphabet)

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("> Initial state probabilities (pi)"):
                i += 2
                pi = []
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(">"):
                    pi.append(float(lines[i].strip().split("\t")[1]))
                    i += 1
                hmm.pi = np.array(pi)
                continue
            elif line.startswith("> Transition matrix (A)"):
                i += 2
                A = []
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(">"):
                    A.append([float(x) for x in lines[i].strip().split("\t")[1:]])
                    i += 1
                hmm.A = np.array(A)
                continue
            elif line.startswith("> Emission matrix (B)"):
                i += 2
                B = []
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(">"):
                    B.append([float(x) for x in lines[i].strip().split("\t")[1:]])
                    i += 1
                hmm.B = np.array(B)
                continue
            i += 1

        return hmm


if __name__ == "__main__":
    # Quick test with the unfair casino toy example used just to confirm the sampler runs 
    np.random.seed(42)

    A_test = np.array([[0.9, 0.1], [0.23, 0.77]])
    B_test = np.array([
        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        [1/10, 1/10, 1/10, 1/10, 1/10, 1/2],
    ])
    sizes_test = np.random.randint(100, high=200, size=50, dtype=int)
    alphabet = "123456"

    sequences, labels = [], []
    for size in sizes_test:
        seq, lab = [], []
        state = np.random.choice(2)
        for _ in range(size):
            seq.append(str(np.random.choice(6, p=B_test[state]) + 1))
            lab.append(state)
            state = np.random.choice(2, p=A_test[state])
        sequences.append("".join(seq))
        labels.append(lab)

    gibbs_casino = HMM_Gibbs(2, alphabet, seed=42)
    encoded = [gibbs_casino.encode(s) for s in sequences]
    gibbs_casino.gibbs(encoded, max_iter=100)

    print("\nLearned A:\n", np.round(gibbs_casino.A, 2))
    print("Learned B:\n", np.round(gibbs_casino.B, 2))

    # TERRIBLE 