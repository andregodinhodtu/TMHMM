
import os
import numpy as np


# module helpers

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


# tiny remove first aminoacid helper
def strip_first_residue(sequences, labels, headers):
    """
    Removes the first position (amino acid + label) from every sequence.
    Used because the initiator methionine is always M and always Inside,
    so it adds no topology information and biases the model.
    """
    stripped_seqs  = [s[1:] for s in sequences]
    stripped_labs  = [l[1:] for l in labels]
    return stripped_seqs, stripped_labs, headers



# shared base class
class HMM: # not usable by itself

    """ 
    Attributes
    ----------
    n_states : int
    alphabet : str
    n_symbols : int
    pi : np.ndarray (n_states,)
    A  : np.ndarray (n_states, n_states)
    B  : np.ndarray (n_states, n_symbols)
    alpha   : np.ndarray (n_states, T)  — set by forward()
    G_alpha : np.ndarray (T,)           — set by forward()
    """


    TRAINING_METHOD = "base"   # overridden by subclasses

    # common methods:  
    # init, scaled forward pass
    # output model, input model, load model

    def __init__(self, number_of_states, alphabet, seed=42):
        """
        Initializes the HMM with random parameters.

        Parameters
        ----------
        number_of_states : int
        alphabet : str
        seed : int, optional
        """
        np.random.seed(seed)

        self.n_states  = number_of_states
        self.alphabet  = alphabet
        self.n_symbols = len(set(alphabet))

        self.pi = np.random.dirichlet(np.ones(self.n_states))

        self.A  = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        self.B  = np.random.dirichlet(np.ones(self.n_symbols), size=self.n_states)

        # set by forward()
        self.alpha   = None
        self.G_alpha = None

        




    def forward(self, input_encode): # stays exactly the same
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


    # MODEL I/O + LOAD

    def output_model(self, output_filename):
        """
        Saves the trained HMM parameters to a file.

        Parameters
        ----------
        output_filename : str
            Path to the output file.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_default_path = repo_root + "/results/models/" + output_filename

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





class HMM_BaumWelch(HMM):
    """
    Hidden Markov Model (HMM) implementation with Baum-Welch training.

    Attributes
    ----------
    Inherits from HMM: __init__, forward(), evaluate(), output_model(),
    input_model(), load_model().

    Additional attributes (set during training)
    -------------------------------------------
    beta   : np.ndarray (n_states, T)  — scaled backward probabilities
    G_beta : np.ndarray (T,)           — backward scaling factors
    gamma  : np.ndarray (n_states, T)  — posterior state probabilities
    xi     : np.ndarray (n_states, n_states, T-1)  — pairwise posteriors
    """

    TRAINING_METHOD = "Baum-Welch"

    def __init__(self, number_of_states, alphabet, seed=42):
        super().__init__(number_of_states, alphabet, seed)

        # Backward algorithm results
        self.beta   = None
        self.G_beta = None

        # Baum-Welch statistics
        self.gamma = None
        self.xi    = None

   

    

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

# keep the old name as an alias so existing notebooks don't break
HMM_efficient = HMM_BaumWelch

# GIBBS SAMPLING CLASS

class HMM_Gibbs(HMM):
    """
    HMM trained with Gibbs sampling.

    Samples one concrete path per sequence from its posterior and re-estimates
    parameters from counts over samples.

    Inherits from HMM: __init__, forward(), output_model(),
    input_model(), load_model().
    """

    TRAINING_METHOD = "Gibbs sampling"

    def _sample_path(self, input_encode):
        """
        Runs forward() to get alpha, then samples a hidden-state path
        backwards from the posterior (forward-filtering / backward-sampling).

        At each t:
            P(z_t | z_{t+1}, obs) ∝ alpha[z_t, t] * A[z_t, z_{t+1}]

        Parameters
        ----------
        input_encode : list of int

        Returns
        -------
        path : np.ndarray (T,)
            One sampled sequence of hidden-state indices.
        """

        T = len(input_encode)
        self.forward(input_encode)      # populates self.alpha, self.G_alpha
        path = np.zeros(T, dtype=int)

        # Sample last state
        probs = self.alpha[:, T - 1]
        probs = probs / probs.sum()
        path[T - 1] = np.random.choice(self.n_states, p=probs)

        # Sample backwards
        for t in range(T - 2, -1, -1):
            probs = self.alpha[:, t] * self.A[:, path[t + 1]]
            probs = probs / probs.sum()
            path[t] = np.random.choice(self.n_states, p=probs)

        return path

    def _update_parameters(self, all_paths, all_encoded):
        """
        Re-estimates pi, A, B by counting over the sampled state paths.

        A small pseudocount (1e-3) is added before normalising so a state
        that wasn't sampled in one iteration doesn't become permanently
        unreachable in the next.

        Parameters
        ----------
        all_paths   : list of np.ndarray — one sampled path per sequence
        all_encoded : list of list of int
        """

        # pi
        pi_counts = np.zeros(self.n_states)
        for path in all_paths:
            pi_counts[path[0]] += 1
        pi_counts += 1e-3
        self.pi = pi_counts / pi_counts.sum()

        # A
        A_counts = np.zeros((self.n_states, self.n_states))
        for path in all_paths:
            for t in range(len(path) - 1):
                A_counts[path[t], path[t + 1]] += 1
        A_counts += 1e-3
        self.A = A_counts / A_counts.sum(axis=1, keepdims=True)

        # B
        B_counts = np.zeros((self.n_states, self.n_symbols))
        for path, enc in zip(all_paths, all_encoded):
            for t, obs in enumerate(enc):
                B_counts[path[t], obs] += 1
        B_counts += 1e-3
        self.B = B_counts / B_counts.sum(axis=1, keepdims=True)

    def gibbs(self, sequences, max_iter=100, epsilon=1e-4):
        """
        Trains the HMM via Gibbs sampling over multiple sequences.

        Parameters
        ----------
        sequences : list of list of int
        max_iter  : int, optional   (default 100)
        epsilon   : float, optional (default 1e-4)

        Returns
        -------
        log_likelihoods : list of float
        """

        self.log_likelihoods = []

        for iteration in range(max_iter):

            # Log-likelihood (requires another forward pass on updated params)
            log_likelihood = 0
            for enc in sequences:
                self.forward(enc)
                log_likelihood += np.sum(np.log(self.G_alpha))

            self.log_likelihoods.append(log_likelihood)
            print(f"Iteration {iteration + 1} | log-likelihood: {log_likelihood:.4f}")

            # Convergence: moving average, because Gibbs LL is not monotonic.
            if iteration >= 10:
                prev_avg = np.mean(self.log_likelihoods[-10:-5])
                curr_avg = np.mean(self.log_likelihoods[-5:])
                if abs(curr_avg - prev_avg) < epsilon:
                    print(f"Converged at iteration {iteration + 1}")
                    break

            # E-step (stochastic): sample one path per sequence
            all_paths = [self._sample_path(enc) for enc in sequences]

            # M-step: count over sampled paths
            self._update_parameters(all_paths, sequences)

        return self.log_likelihoods



# TINY TEST SECTION
def parse_toy_rolls(text, alphabet="123456"):
    cleaned = text.replace(" ", "").replace(",", "")
    return encode(cleaned, alphabet)


if __name__ == "__main__":
    alphabet = "123456"

    train_sequences = [
        parse_toy_rolls("1 2 3 6 6 6 6 6", alphabet),
        parse_toy_rolls("6 6 6 6 2 3 1", alphabet),
        parse_toy_rolls("1,2,6,6,6,6,6,6", alphabet),
    ]

    print("=== Baum-Welch ===")
    bw = HMM_BaumWelch(number_of_states=2, alphabet=alphabet, seed=0)
    bw.baum_welch(train_sequences, max_iter=5)
    print("pi:", bw.pi)
    print("A:", bw.A)
    print("B:", bw.B)

    print("\n=== Gibbs ===")
    gibbs = HMM_Gibbs(number_of_states=2, alphabet=alphabet, seed=0)
    gibbs.gibbs(train_sequences, max_iter=5)
    print("pi:", gibbs.pi)
    print("A:", gibbs.A)
    print("B:", gibbs.B)





