import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser(description="SMM with Monte Carlo Method")
parser.add_argument("-train", action="store", dest="TRAIN_FILE", type=str, help="Train dataset")
args = parser.parse_args()

train_file = args.TRAIN_FILE

# emission_counts[state][amino_acid]
emission_counts = defaultdict(lambda: defaultdict(int))
all_states = set()
all_aas = set() # To keep track of all 20 amino acids found

with open(train_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]

for i in range(0, len(lines), 3):
    if i + 2 < len(lines):
        aa_seq = lines[i+1]    # The Amino Acid Sequence (Observations)
        state_seq = lines[i+2] # The State Sequence (Hidden States)
        
        # zip() lets us loop through both sequences at the exact same time
        for aa, state in zip(aa_seq, state_seq):
            emission_counts[state][aa] += 1
            
            all_states.add(state)
            all_aas.add(aa)

# Convert sets to sorted lists for clean DataFrame headers
states_list = sorted(list(all_states))
aa_list = sorted(list(all_aas))

# Rows = Hidden States, Columns = Observable Amino Acids
count_matrix = pd.DataFrame(0, index=states_list, columns=aa_list, dtype=float)

# Populate the matrix with raw counts
for state in states_list:
    for aa in aa_list:
        count_matrix.at[state, aa] = emission_counts[state][aa]

# --- PSEUDOCOUNTS ---
# Add a microscopic number so rare AAs don't get permanently locked at 0.0 in Baum-Welch
pseudocount = 1e-6
smoothed_counts = count_matrix + pseudocount

# Normalize rows so that the probabilities of all AAs emitted by a state equal 1.0
probability_matrix = smoothed_counts.div(smoothed_counts.sum(axis=1), axis=0)

# Display the results
print("=== EMPIRICAL EMISSION PROBABILITY MATRIX ===")
print(probability_matrix.round(4))