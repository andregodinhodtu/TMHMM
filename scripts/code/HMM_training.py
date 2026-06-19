import numpy as np
from datetime import datetime

def process_symbols(symbols):
    """
    Returns a sorted array of unique symbols from the input string.
    """
    return np.array(sorted(set(symbols)))

def encode(sequence,alphabet):
        """
        Encodes a sequence of characters into a list of integer indices
        based on their position in the alphabet.

        Parameters
        ----------
        sequence : str
            The input sequence to encode (e.g. an amino acid sequence "ACMKL").

        Returns
        -------
        list of int
            A list of integers where each element is the index of the
            corresponding character in the alphabet.

        Example
        -------
        >>> hmm = HMM(n_states=3, alphabet="ACDEFGHIKLMNPQRSTVWY")
        >>> hmm.encode("ACD")
        [0, 1, 2]
        """

        char_to_idx = {char: i for i, char in enumerate(alphabet)}

        try:
            enc = [char_to_idx[char] for char in sequence]
        except KeyError as e:
            raise ValueError(f"Sequence contains character not found in alphabet: {e}")

        return enc

def parse_fasta(filepath, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    """
    Parses a DeepTMHMM fasta file into sequences and labels.

    Parameters
    ----------
    filepath : str
        Path to the fasta file.
    alphabet : str
        Valid amino acid symbols. Sequences containing other characters
        (e.g. X) will be filtered out.

    Returns
    -------
    sequences : list of list of int
        Encoded amino acid sequences.
    labels : list of list of int
        Corresponding encoded state label sequences.
    headers : list of str
        Protein identifiers.
    """

    sequences = []
    labels = []
    headers = []

    label_map = {"S": 0, "I": 1, "M": 2, "B": 3, "P": 4, "O": 5}
    char_to_idx = {char: i for i, char in enumerate(alphabet)}

    with open(filepath, "r") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            header = lines[i][1:]  # strip >
            seq = lines[i+1].strip()
            lab = lines[i+2].strip()

            # filter out sequences with unknown characters
            if all(c in alphabet for c in seq) and len(seq) == len(lab):
                headers.append(header)
                sequences.append([char_to_idx[c] for c in seq])
                labels.append([label_map[c] for c in lab])

            i += 3
        else:
            i += 1

    print(f"Parsed {len(sequences)} valid sequences")
    return sequences, labels, headers

class HMM_efficient():
    """
    Hidden Markov Model (HMM) implementation with Baum-Welch training.

    Attributes
    ----------
    n_states : int
        Number of hidden states in the model.
    alphabet : str
        String of valid emission symbols (e.g. "ACDEFGHIKLMNPQRSTVWY").
    n_symbols : int
        Number of unique symbols in the alphabet.
    pi : np.ndarray of shape (n_states,)
        Initial state probability vector.
    A : np.ndarray of shape (n_states, n_states)
        Transition probability matrix, where A[i][j] is the probability
        of transitioning from state i to state j.
    B : np.ndarray of shape (n_states, n_symbols)
        Emission probability matrix, where B[i][k] is the probability
        of emitting symbol k in state i.
    alpha : np.ndarray of shape (n_states, sequence_length)
        The scaled forward probabilities.
    G_alpha : np.ndarray of shape (sequence_length,)
        The scaling factors at each position t of the forward algorithm.
    beta : np.ndarray of shape (n_states, sequence_length)
        The scaled backward probabilities.
    G_beta : np.ndarray of shape (sequence_length,)
        The scaling factors at each position t of the backward algorithm.
    """

    def __init__(self, number_of_states, alphabet, seed=42):
        """
        Initializes the HMM with random parameters.

        Parameters
        ----------
        number_of_states : int
            Number of hidden states in the model.
        alphabet : str
            String of valid emission symbols.
        seed : int, optional
            Random seed for reproducibility (default: 42).
        """

        np.random.seed(seed)

        self.n_states = number_of_states
        self.alphabet = alphabet
        self.n_symbols = len(set(alphabet))

        # Initialize and normalize pi
        self.pi = np.random.dirichlet(np.ones(self.n_states))

        # Initialize and normalize transition matrix A
        self.A = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)

        # Initialize and normalize emission matrix B
        self.B = np.random.dirichlet(np.ones(self.n_symbols), size=self.n_states)

        # Forward algorithm results
        self.alpha = None
        self.G_alpha = None

        # Backward algorithm results
        self.beta = None
        self.G_beta = None

        # Baum-Welch training
        self.gamma = None
        self.xi = None

    def forward(self, input_encode):
        """
        Runs the forward algorithm, filling the alpha matrix with scaled
        forward probabilities using the recurrence:

            alpha[:, 0] = pi * B[:, y_0]
            alpha[:, t] = B[:, y_t] * (A.T @ alpha[:, t-1])

        All columns are scaled by G at each step to avoid numerical underflow.
        Results are stored in self.alpha and self.G_alpha.

        Parameters
        ----------
        input_encode : list of int
            The encoded input sequence (output of encode()).
        """

        T = len(input_encode)
        self.alpha = np.zeros((self.n_states, T))
        self.G_alpha = np.zeros(T)

        # t=0
        self.alpha[:, 0] = self.pi * self.B[:, input_encode[0]]
        self.G_alpha[0] = self.alpha[:, 0].sum()
        self.alpha[:, 0] /= self.G_alpha[0]

        # t=1 to T-1
        for t in range(1, T):
            self.alpha[:, t] = self.B[:, input_encode[t]] * (self.A.T @ self.alpha[:, t-1])
            # Rescaling (divide by sum of values of all states in current t)
            self.G_alpha[t] = self.alpha[:, t].sum()
            self.alpha[:, t] /= self.G_alpha[t]

    def backward(self, input_encode):
        """
        Runs the backward algorithm, filling the beta matrix with scaled
        backward probabilities using the recurrence:

            beta[:, T-1] = 1
            beta[:, t]   = A @ (B[:, y_{t+1}] * beta[:, t+1])

        All columns are scaled by G at each step to avoid numerical underflow.
        Results are stored in self.beta and self.G_beta.

        Parameters
        ----------
        input_encode : list of int
            The encoded input sequence (output of encode()).
        """

        T = len(input_encode)
        self.beta = np.zeros((self.n_states, T))
        self.G_beta = np.zeros(T)

        # t=T-1: boundary condition
        self.beta[:, -1] = 1

        # t=T-2 to 0
        for t in range(T - 2, -1, -1):
            self.beta[:, t] = self.A @ (self.B[:, input_encode[t+1]] * self.beta[:, t+1])
            # Rescaling (divide by sum of values of all states in current t)
            self.G_beta[t] = self.beta[:, t].sum()
            self.beta[:, t] /= self.G_beta[t]

    def compute_gamma(self):
        """
        Computes the posterior probabilities (gamma) for each state at each
        position in the sequence.

        gamma[i][t] is the probability of being in state i at time t given
        the full observation sequence:

            gamma[:, t] = (alpha[:, t] * beta[:, t]) / sum_j(alpha[j][t] * beta[j][t])

        Results are stored in self.gamma.
        Must call forward() and backward() before this method.
        """

        # Brand new vector created just pairwise multipilication
        self.gamma = self.alpha * self.beta
        self.gamma /= self.gamma.sum(axis=0, keepdims=True)


    def compute_xi(self, input_encode):
        """
        Computes the pairwise posterior probabilities (xi) for each pair of
        states at each position in the sequence.

        xi[i][j][t] is the probability of being in state i at time t and
        transitioning to state j at time t+1, given the full observation sequence:

            xi[:, :, t] = outer(alpha[:, t], B[:, y_{t+1}] * beta[:, t+1]) * A
                          normalized by its sum

        Results are stored in self.xi.
        Must call forward() and backward() before this method.

        Parameters
        ----------
        input_encode : list of int
            The encoded input sequence (output of encode()).
        """

        T = self.alpha.shape[1]
        self.xi = np.zeros((self.n_states, self.n_states, T - 1))

        for t in range(T - 1):
            # outer product: alpha[:, t] (n_states,) x (B[:, y_{t+1}] * beta[:, t+1]) (n_states,)
            self.xi[:, :, t] = np.outer(self.alpha[:, t], self.B[:, input_encode[t+1]] * self.beta[:, t+1]) * self.A
            self.xi[:, :, t] /= self.xi[:, :, t].sum()

    def _update_parameters(self, all_gamma, all_xi, all_encoded):
        """
        Updates the HMM parameters pi, A, and B using the Baum-Welch
        update equations across multiple sequences.

        Parameters
        ----------
        all_gamma : list of np.ndarray
            List of gamma matrices, one per sequence.
        all_xi : list of np.ndarray
            List of xi arrays, one per sequence.
        all_encoded : list of list of int
            List of encoded sequences.
        """

        R = len(all_encoded)

        # Update pi: average of gamma at t=0 across all sequences
        self.pi = np.sum([all_gamma[r][:, 0] for r in range(R)], axis=0) / R

        # Update A: sum of xi over time and sequences, normalize each row
        A_num = np.sum([all_xi[r].sum(axis=2) for r in range(R)], axis=0)
        self.A = A_num / A_num.sum(axis=1, keepdims=True)

        # Update B: expected emissions per state
        B_num = np.zeros((self.n_states, self.n_symbols))
        for r in range(R):
            T = len(all_encoded[r])
            for k in range(self.n_symbols):
                mask = np.array(all_encoded[r]) == k
                B_num[:, k] += all_gamma[r][:, mask].sum(axis=1)

        self.B = B_num / B_num.sum(axis=1, keepdims=True)


    def baum_welch(self, sequences, max_iter=100, epsilon=1e-4):
        """
        Trains the HMM using the Baum-Welch algorithm over multiple sequences.

        Parameters
        ----------
        sequences : list of list of int
            List of encoded sequences to train on.
        max_iter : int, optional
            Maximum number of iterations (default: 100).
        epsilon : float, optional
            Convergence threshold for log-likelihood (default: 1e-4).

        Returns
        -------
        log_likelihoods : list of float
            Log-likelihood at each iteration, useful for diagnostics.
        """

        self.log_likelihoods = []

        for iteration in range(max_iter):
            all_gamma = []
            all_xi = []
            log_likelihood = 0

            # E-step: forward, backward, gamma, xi for each sequence
            for enc in sequences:
                self.forward(enc)
                self.backward(enc)
                self.compute_gamma()
                self.compute_xi(enc)

                all_gamma.append(self.gamma.copy())
                all_xi.append(self.xi.copy())
                log_likelihood += np.sum(np.log(self.G_alpha))

            self.log_likelihoods.append(log_likelihood)
            print(f"Iteration {iteration + 1} | log-likelihood: {log_likelihood:.4f}")

            # Check convergence
            if iteration > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < epsilon:
                print(f"Converged at iteration {iteration + 1}")
                break

            # M-step: update parameters
            self._update_parameters(all_gamma, all_xi, sequences)

        return self.log_likelihoods

    def input_model(self, input_default_path):
        """
        Loads a trained HMM model from a file saved by output_model().

        Parameters
        ----------
        input_filename : str
            Path to the model file.
        """

        with open(input_default_path, "r") as f:
            # Reading all to memory because it's small
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("> Alphabet:"):
                self.alphabet = line.split(": ")[1]
                self.n_symbols = len(set(self.alphabet))

            elif line.startswith("> Number of hidden states:"):
                self.n_states = int(line.split(": ")[1])

            elif line.startswith("> Initial state probabilities (pi)"):
                i += 2  # skip shape line
                pi = []
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(">"):
                    val = float(lines[i].strip().split("\t")[1])
                    pi.append(val)
                    i += 1
                self.pi = np.array(pi)
                continue

            elif line.startswith("> Transition matrix (A)"):
                i += 2  # skip shape line
                A = []
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(">"):
                    row = [float(x) for x in lines[i].strip().split("\t")[1:]]
                    A.append(row)
                    i += 1
                self.A = np.array(A)
                continue

            elif line.startswith("> Emission matrix (B)"):
                i += 2  # skip shape line
                B = []
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(">"):
                    row = [float(x) for x in lines[i].strip().split("\t")[1:]]
                    B.append(row)
                    i += 1
                self.B = np.array(B)
                continue

            i += 1

    @classmethod
    def load_model(cls, input_filename):
        """
        Loads a trained HMM from a file and returns a new HMM instance.

        Parameters
        ----------
        input_filename : str
            Path to the model file.

        Returns
        -------
        HMM_efficient
            A new HMM instance with the loaded parameters.
        """

        input_default_path = "../../results/models/" + input_filename

        with open(input_default_path, "r") as f:
            lines = f.readlines()

        # parse alphabet and n_states first to initialize
        alphabet = None
        n_states = None
        for line in lines:
            if line.startswith("> Alphabet:"):
                alphabet = line.split(": ")[1].strip()
            if line.startswith("> Number of hidden states:"):
                n_states = int(line.split(": ")[1].strip())
            if alphabet and n_states:
                break

        # create instance without random init
        hmm = cls(n_states, alphabet)

        # then parse and overwrite parameters
        hmm.input_model(input_default_path)

        return hmm

    def output_model(self, output_filename):
        """
        Saves the trained HMM parameters to a file.

        Parameters
        ----------
        output_filename : str
            Path to the output file.
        """

        output_default_path = "../../results/models/" + output_filename

        with open(output_default_path, "w") as f:
            f.write(f"> Hidden Markov Model: {output_filename}\n")
            f.write(f"> Training method: Baum-Welch\n")
            f.write(f"> Alphabet: {self.alphabet}\n")
            f.write(f"> Number of hidden states: {self.n_states}\n")
            f.write(f"> Number of symbols: {self.n_symbols}\n")
            f.write("\n")

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


