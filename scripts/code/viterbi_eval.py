#!/usr/bin/env python
# coding: utf-8

# # Viterbi
# Implementation of the Viterbi algorithm.
# 
# 

# ## Python Imports

# In[3]:


import numpy as np
import os
from argparse import ArgumentParser
from HMM_training import HMM_efficient


# ## Parser

# In[4]:


parser = ArgumentParser(description="Using the Viterbi algorithm to evaluate a HMM model with already annotated data", add_help=True)
parser.add_argument("-eval", action="store", dest="evaluation_file", type=str, help="Input evaluation file")
parser.add_argument("-hmm", action="store", dest="hmm_file", type=str, help="HMM model file (without path as it is already included in the HMM class)")
parser.add_argument("-attrib", action="store_true", dest="attribute_states", help="Mode to label states of transition matrix")
parser.add_argument("-o", action="store", dest="output_dir", type=str, default=".", help="Path to the output directory")

args = parser.parse_args()
evaluation_file = args.evaluation_file
hmm_file = args.hmm_file
attribute_states = args.attribute_states
output_dir = args.output_dir


# ## Import data

# In[ ]:


def load_sequences(filepath):
    sequences, ids, topologies = [], [], []

    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines), 3):
        id_line = lines[i]
        sequence = lines[i+1]
        topology = lines[i+2]

        protein_id = id_line.split(">")[1]

        ids.append(protein_id)
        sequences.append(sequence)
        topologies.append(topology)

    return sequences, ids, topologies


# In[ ]:


eval_sequences, id_sequences, label_sequences = load_sequences(evaluation_file)
hmm_model = HMM_efficient.load_model(hmm_file)

#adapt the load function and type in function of the output of the Baum Welsh algorithm
transition_matrix = hmm_model.A
emission_probs = hmm_model.B
initial_prob = hmm_model.pi


# ## Simulation Main

# ### Parameters

# In[21]:


def initialize(encode_sequence, states, initial_prob, transition_matrix, emission_probs):

    delta = np.zeros(shape=(states, len(encode_sequence)))

    arrows = np.ndarray(shape=(states, len(encode_sequence)), dtype=object)

    # initial conditions
    for i in range(0, states):

        delta[i][0] = initial_prob[i] + emission_probs[i][encode_sequence[0]] # Remember we work in log space 

        arrows[i][0] = 0

    return delta, arrows


# ## Encode sequence as integers (index values)

# In[2]:


def encode( sequence, symbols):

    enc = [0] * len(sequence)

    for i in range(len(sequence)):
        enc[i] = symbols.find(sequence[i])

    return(enc)


# ## Viterbi Function

# In[ ]:


def viterbi(input_encode, states, initial_prob, transition_matrix, emission_probs):

    delta, arrows = initialize(input_encode, states, initial_prob, transition_matrix, emission_probs)

    for i in range(1, len(input_encode)):

            for j in range(0, states):

                max_arrow_prob = -np.inf # A very low negative number
                max_arrow_prob_state = -1

                for k in range(0, states):

                    # arrow_prob is the probability of ending in the state j from the state k
                    arrow_prob = transition_matrix[k, j] + delta[k][i-1]

                    if arrow_prob > max_arrow_prob: 
                        max_arrow_prob = arrow_prob
                        max_arrow_prob_state = k

                # store prob
                delta[j][i] = emission_probs[j][input_encode[i]] + max_arrow_prob

                # store arrow
                arrows[j][i] = max_arrow_prob_state

    return(delta, arrows)


# ## Backtracking function for optimal path

# In[ ]:


def optimal_path(input_sequence, sequence_id, delta, arrows, attribute_states=False):
    path = []

    max_state = np.argmax(delta[:, -1]) # Find the index of the max value in the last column of delta
    max_value = delta[max_state, -1] # Find the max value in the last column of delta

    path.append(str(max_state))

    old_state = int(max_state)

    for i in range(len(input_sequence)-2, -1, -1):

        current_state = int(arrows[old_state][i+1])      
        path.append(str(current_state))  
        old_state = current_state 

    if attribute_states:
        # preparing the output
        path_str = "".join(reversed(path))

        output_dict = {
            "sequence_id": sequence_id,
            "sequence": input_sequence,
            "encoded_path": path_str,
            "log_probability": max_value
        }

    else:
        # preparing the output
        path_str = "".join(reversed(path))
        translate_state = str.maketrans("012", "IMO") # Modify in function of the states
        final_path = path_str.translate(translate_state)

        output_dict = {
            "sequence_id": sequence_id,
            "encoded_sequence": input_sequence,
            "encoded_path": path_str,
            "predicted_path": final_path,
            "log_probability": max_value
        }

    return(output_dict)


# ## Initialization
#  

# In[1]:


states = np.shape(transition_matrix)[0]

symbols = "ACDEFGHIKLMNPQRSTVWY"
nsymbols = len(symbols)

#Defining model: directly in log space for Viterbi

epsilon = 1e-10

# Safe conversion to log space to avoid log(0)
log_transition_matrix = np.log10(np.where(transition_matrix > 0, transition_matrix, epsilon))
log_emission_probs = np.log10(np.where(emission_probs > 0, emission_probs, epsilon))
log_initial_prob = np.log10(np.where(initial_prob > 0, initial_prob, epsilon))


# ## Main Loop

# In[ ]:


if attribute_states:

    random_idx = np.random.randint(0, len(eval_sequences), 10)
    for idx in random_idx:
        seq_id = id_sequences[idx]
        sequence = eval_sequences[idx]
        input_encode = encode(sequence, symbols)
        delta, arrows = viterbi(input_encode, states, log_initial_prob, log_transition_matrix, log_emission_probs)
        seq_summary = optimal_path(sequence, seq_id, delta, arrows, attribute_states)
        print("Sequence ", seq_id)
        print("log(Max_path):", seq_summary["log_probability"])
        print("Seq:", sequence)
        print("Encoded path:", seq_summary["encoded_path"])
        print("Actual labels:", label_sequences[idx])

else:
    all_results = []
    for (i, sequence) in enumerate(eval_sequences):
        seq_id = id_sequences[i]
        input_encode = encode(sequence, symbols)
        delta, arrows = viterbi(input_encode, states, log_initial_prob, log_transition_matrix, log_emission_probs)
        seq_summary = optimal_path(sequence, seq_id, delta, arrows)
        all_results.append(seq_summary)



# ## Evaluation of the HMM model

# In[ ]:


if not attribute_states:
    total_amino_acids = 0
    global_correct_predictions = 0

    # Store the accuracy of each individual sequence
    detailed_metrics = []

    for i, result in enumerate(all_results):
        predicted_path = result["predicted_path"]
        true_path = label_sequences[i]

        # Sanity check: Ensure paths have the exact same length
        if len(predicted_path) != len(true_path):
            print(f"Warning: Length mismatch for sequence {result['sequence_id']}")
            continue

        seq_correct = sum(1 for p, t in zip(predicted_path, true_path) if p == t)
        seq_length = len(predicted_path)

        # Calculate sequence-level accuracy
        seq_accuracy = seq_correct / seq_length

        # Update global counters
        total_amino_acids += seq_length
        global_correct_predictions += seq_correct

        # Save detailed results
        detailed_metrics.append({
            "sequence_id": result["sequence_id"],
            "accuracy": seq_accuracy
        })

    # Calculate global accuracy across the entire dataset
    global_accuracy = global_correct_predictions / total_amino_acids if total_amino_acids > 0 else 0


# ## Creating output files

# In[ ]:


if not attribute_states:
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "viterbi_predictions_detailed.txt")

    with open(output_filename, "w") as out_file:
        out_file.write("### VITERBI EVALUATION RESULTS ###\n")
        out_file.write(f"Global Accuracy: {global_accuracy * 100:.2f}%\n")
        out_file.write("=" * 50 + "\n\n")

        for i, result in enumerate(all_results):
            seq_id = result["sequence_id"]
            aa_seq = result["encoded_sequence"] 
            pred_path = result["predicted_path"]
            log_prob = result["log_probability"]
            true_path = label_sequences[i]

            acc = detailed_metrics[i]["accuracy"]

            out_file.write(f">{seq_id} | log_prob: {log_prob:.2f} | accuracy: {acc * 100:.2f}%\n")
            out_file.write(f"SEQ:  {aa_seq}\n")
            out_file.write(f"TRUE: {true_path}\n")
            out_file.write(f"PRED: {pred_path}\n\n")

    print("\n--- Evaluation Complete ---")
    print(f"Global Accuracy: {global_accuracy * 100:.2f}%")
    print(f"Detailed results successfully saved to: {output_filename}")

