import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser(description="Compute initial state frequencies from position 0 and 1")
parser.add_argument("-train", action="store", dest="TRAIN_FILE", type=str, help="Train dataset")
args = parser.parse_args()
train_file = args.TRAIN_FILE

# count states at position 0 and position 1
position_0_counts = defaultdict(int)
position_1_counts = defaultdict(int)
all_states = set()

with open(train_file, 'r') as file:
    lines = [line.strip() for line in file.readlines()]

for i in range(0, len(lines), 3):
    if i + 2 < len(lines):
        if not lines[i].startswith(">"):
            continue
        
        state_seq = lines[i+2]

        if len(state_seq) > 0:
            state_0 = state_seq[0]
            if state_0 != 'P':
                position_0_counts[state_0] += 1
                all_states.add(state_0)

        if len(state_seq) > 1:
            state_1 = state_seq[1]
            if state_1 != 'P':
                position_1_counts[state_1] += 1
                all_states.add(state_1)

states_list = sorted(list(all_states))

# normalize to probabilities
total_0 = sum(position_0_counts.values())
total_1 = sum(position_1_counts.values())

print("=== INITIAL STATE FREQUENCIES AT POSITION 0 ===")
for state in states_list:
    count = position_0_counts[state]
    print(f"  {state}: {count:4d} ({100*count/total_0:.2f}%)")

print("\n=== INITIAL STATE FREQUENCIES AT POSITION 1 ===")
for state in states_list:
    count = position_1_counts[state]
    print(f"  {state}: {count:4d} ({100*count/total_1:.2f}%)")

# build pi vectors
pi_0 = {state: position_0_counts[state] / total_0 for state in states_list}
pi_1 = {state: position_1_counts[state] / total_1 for state in states_list}

print("\npi from position 0:", {k: round(v, 4) for k, v in pi_0.items()})
print("pi from position 1:", {k: round(v, 4) for k, v in pi_1.items()})