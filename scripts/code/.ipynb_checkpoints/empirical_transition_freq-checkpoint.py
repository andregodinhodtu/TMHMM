import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser(description="SMM with Monte Carlo Method")
parser.add_argument("-train", action="store", dest="TRAIN_FILE", type=str, help="Train dataset")

args = parser.parse_args()
train_file = args.TRAIN_FILE

# transition_counts[current_state][next_state]
transition_counts = defaultdict(lambda: defaultdict(int))
all_states = set()

with open(train_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]

for i in range(0, len(lines), 3):
    if i + 2 < len(lines):
        state_seq = lines[i+2] 
        
        for j in range(len(state_seq) - 1):
            current_state = state_seq[j]
            next_state = state_seq[j+1]
            
            # --- LOGIC: Ignore any transition involving 'P' ---
            if current_state == 'P' or next_state == 'P':
                continue
            # --------------------------------------------------
            
            transition_counts[current_state][next_state] += 1
            
            all_states.add(current_state)
            all_states.add(next_state)

states_list = sorted(list(all_states))

# Rows = Current State, Columns = Next State
count_matrix = pd.DataFrame(0, index=states_list, columns=states_list, dtype=float)

for s1 in states_list:
    for s2 in states_list:
        count_matrix.at[s1, s2] = transition_counts[s1][s2]

# --- PSEUDOCOUNTS ---
# Add a microscopic number so transitions don't get permanently locked at 0.0 in Baum-Welch
pseudocount = 1e-6
smoothed_counts = count_matrix + pseudocount

# Because 'P' was never added to the matrix, the sum(axis=1) automatically ignores it!
probability_matrix = smoothed_counts.div(smoothed_counts.sum(axis=1), axis=0)

# Display the results
print("=== EMPIRICAL TRANSITION PROBABILITY MATRIX (EXCLUDING 'P') ===")
print(probability_matrix.round(4))