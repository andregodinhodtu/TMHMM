# loaderss :)
import numpy as np

def parse_fasta(filepath, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    """
    Parses a DeepTMHMM fasta file into sequences and labels.

    Parameters
    ----------
    filepath : str
        Path to the fasta file.
    alphabet : str
        Valid amino acid symbols. 
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

    label_map = {"S": 0, "I": 1, "M": 2, "O": 5}
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



def load_empirical_transition(filepath):
    """
    Reads an empirical transition probability matrix from a file.

    Parameters
    ----------
    filepath : str
        Path to the empirical transition probability file.

    Returns
    -------
    A : np.ndarray of shape (n_states, n_states)
        Transition probability matrix.
    states : list of str
        State labels in order.
    """

    states = []
    rows = []

    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    for line in lines:
        # skip header lines and empty lines
        if line.startswith("===") or line == "":
            continue

        tokens = line.split()

        # skip column header line — all tokens are state labels (no floats)
        try:
            float(tokens[1])
        except (ValueError, IndexError):
            continue

        # valid data row
        if tokens[0] in ["I", "M", "O", "S", "B", "P"]:
            states.append(tokens[0])
            rows.append([float(x) for x in tokens[1:]])
    
    A = np.array(rows)
    A += 1e-3
    A /= A.sum()

    print(f"States: {states}")
    print(f"Shape:  {A.shape}")
    print(f"Matrix:\n{np.round(A, 4)}")

    return A, states


def load_empirical_emission(filepath):

    states = []
    rows = []
    amino_acids = []

    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    for line in lines:
        if line.startswith("===") or line == "":
            continue

        tokens = line.split()

        if not tokens:
            continue

        # try to parse second token as float
        # if it works → data row, if not → header row
        try:
            float(tokens[1])
            # it's a data row
            if tokens[0] in ["I", "M", "O", "S", "B", "P"]:
                states.append(tokens[0])
                rows.append([float(x) for x in tokens[1:] if x != "..."])

        except (ValueError, IndexError):
            # it's a header row — extract amino acids
            if len(amino_acids) == 0:
                amino_acids = [t for t in tokens if t != "..."]

    B = np.array(rows)
    B = B + 1e-3
    B /= B.sum()
    
    print(f"States:      {states}")
    print(f"Amino acids: {amino_acids}")
    print(f"Shape:       {B.shape}")

    return B, states, amino_acids



def load_empirical_init_prob(filepath, state_order=["I", "M", "O", "S"]):
    """
    Loads empirical initial state probabilities from file.

    Parameters
    ----------
    filepath : str
        Path to the empirical init prob file.
    state_order : list of str
        Order of states to match HMM state indices.

    Returns
    -------
    pi : np.ndarray of shape (n_states,)
        Initial state probability vector.
    """

    pi_dict = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    # find the "pi from position 1" line
    for line in lines:
        if line.startswith("pi from position 1:"):
            # parse the dict string
            dict_str = line.split(":", 1)[1].strip()
            pi_dict = eval(dict_str)
            break

    # build pi array in correct state order
    pi = np.array([pi_dict.get(state, 0.0) for state in state_order])

    pi = pi + 1e-3
    # normalize
    pi /= pi.sum()



    print(f"State order: {state_order}")
    print(f"pi: {np.round(pi, 4)}")

    return pi


def load_empirical_params(transition_path, emission_path, init_path):
    """
    Loads empirical pi, A, B directly from files with no reordering.
    State index alignment doesn't matter since training is unsupervised
    and Viterbi + permutation search handles label matching at evaluation.
    """
    A,  _           = load_empirical_transition(transition_path)
    B,  _, _        = load_empirical_emission(emission_path)
    pi              = load_empirical_init_prob(init_path)

    return pi, A, B