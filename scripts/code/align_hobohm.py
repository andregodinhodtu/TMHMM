
# LOAD DATA
import numpy as np
from argparse import ArgumentParser

#AA
alphabet_file = "../../matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)

#blosum50: dict in a dict
blosum_file = "../../matrices/BLOSUM50"
_blosum50 = np.loadtxt(blosum_file, dtype=int).T
blosum50 = {}
for i, letter_1 in enumerate(alphabet):
    blosum50[letter_1] = {}
    for j, letter_2 in enumerate(alphabet):
        blosum50[letter_1][letter_2] = _blosum50[i, j]


def load_sequences(filepath):
    sequences, ids, topologies = [], [], []
    
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]
    
   # skipped = 0
    # 1 entry = 3 lines
    for i in range(0, len(lines), 3):
        id_line = lines[i]
        sequence = lines[i+1]
        topology = lines[i+2]

        if 'X' in sequence:
            #skipped += 1
            continue
        
        protein_id = id_line.split(">")[1].split("|")[0]
        
        ids.append(protein_id)
        sequences.append(sequence)
        topologies.append(topology)
    
    return sequences, ids, topologies 


# ALIGNMENT
def smith_waterman(query, database, scoring_scheme, gap_open, gap_extension):

    P_matrix, Q_matrix, D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension)
    aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, query, database, i_max, j_max)

    return aligned_query, aligned_database, matches


def smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension):

    # Matrix dimensions
    M = len(query)
    N = len(database)

    # D[i,j]: best alignment score ending at position (i,j)
    D_matrix = np.zeros((M+1, N+1), dtype=float)

    # P[i,j]: best score ending with a gap in the database (gap in column)
    P_matrix = np.full((M+1, N+1), -np.inf)

    # Q[i,j]: best score ending with a gap in the query (gap in row)
    Q_matrix = np.full((M+1, N+1), -np.inf)

    # E[i,j]: traceback direction
    #   0 = stop, 1 = diagonal (match/mismatch),
    #   2 = gap in database (from P), 
    #   3 = gap in query (from Q)
    E_matrix = np.zeros((M+1, N+1), dtype=int)

    D_matrix_max_score = 0.0
    D_matrix_i_max = 0
    D_matrix_j_max = 0

    for i in range(1, M+1):
        for j in range(1, N+1):

            # P_matrix[i,j]: gap in database (consume query residue i, hold j)
            gap_open_database = D_matrix[i-1, j] + gap_open
            gap_extension_database = P_matrix[i-1, j] + gap_extension
            P_matrix[i, j] = max(gap_open_database, gap_extension_database)

            # Q_matrix[i,j]: gap in query (consume database residue j, hold i)
            gap_open_query = D_matrix[i, j-1] + gap_open
            gap_extension_query = Q_matrix[i, j-1] + gap_extension
            Q_matrix[i, j] = max(gap_open_query, gap_extension_query)

            # Diagonal score
            diagonal_score = D_matrix[i-1, j-1] + scoring_scheme[query[i-1]][database[j-1]]

            # E_matrix[i,j]: best direction (we do not need 5 scores)
            candidates = [(1, diagonal_score),
                          (2, P_matrix[i, j]),
                          (3, Q_matrix[i, j]),
                          (0, 0.0)]

            direction, max_score = max(candidates, key=lambda x: x[1])

            D_matrix[i, j] = max(0.0, max_score)
            E_matrix[i, j] = direction if max_score > 0 else 0

            # Track global max
            if D_matrix[i, j] > D_matrix_max_score:
                D_matrix_max_score = D_matrix[i, j]
                D_matrix_i_max = i
                D_matrix_j_max = j

    return P_matrix, Q_matrix, D_matrix, E_matrix, D_matrix_i_max, D_matrix_j_max, D_matrix_max_score


def smith_waterman_traceback(E_matrix, query, database, i_max, j_max):

    aligned_query = []
    aligned_database = []
    matches = 0

    i, j = i_max, j_max
    while i > 0 and j > 0:

        # E[i,j] = 0, stop traceback
        if E_matrix[i, j] == 0:
            break

        # E[i,j] = 1, match/mismatch
        elif E_matrix[i, j] == 1:
            aligned_query.append(query[i-1])
            aligned_database.append(database[j-1])
            if query[i-1] == database[j-1]:
                matches += 1
            i -= 1
            j -= 1

        # E[i,j] = 2, gap in database
        elif E_matrix[i, j] == 2:
            aligned_query.append(query[i-1])
            aligned_database.append("-")
            i -= 1

        # E[i,j] = 3, gap in query
        elif E_matrix[i, j] == 3:
            aligned_query.append("-")
            aligned_database.append(database[j-1])
            j -= 1

    aligned_query = "".join(reversed(aligned_query))
    aligned_database = "".join(reversed(aligned_database))

    return aligned_query, aligned_database, matches


def percent_identity(aligned_query, aligned_database, matches):
    aln_len = len(aligned_query)
    if aln_len == 0:
        return 0.0
    return matches / aln_len


# HOBOHM 1
def homology_function(alignment_length, matches):
    homology_score = 2.9 * np.sqrt(alignment_length)
    if matches > homology_score: 
        return "discard", homology_score
    else:
        return "keep", homology_score

def run_hobohom1(seq_type):
    filename = "../../data/raw_data/" + seq_type + "_sequences.3line"
    # load list
    candidate_sequences, candidate_ids, candidate_topologies = load_sequences(filename)
    print("# Number of elements:", len(candidate_sequences))

    accepted_sequences, accepted_ids, accepted_topologies = [], [], []
    accepted_sequences.append(candidate_sequences[0])
    accepted_ids.append(candidate_ids[0])
    accepted_topologies.append(candidate_topologies[0])

    print("# Unique.", 0, len(accepted_sequences)-1, accepted_ids[0])

    # parameters
    scoring_scheme = blosum50
    gap_open = -11
    gap_extension = -1

    for i in range(1, len(candidate_sequences)):
        is_unique = True
        for j in range(0, len(accepted_sequences)):
            
            query = candidate_sequences[i]
            database = accepted_sequences[j]
            
            aligned_query, aligned_database, matches = smith_waterman(query, database, scoring_scheme, gap_open, gap_extension)
            alignment_length = len(aligned_query)
            homology_outcome, homology_score = homology_function(alignment_length, matches)
        
            if homology_outcome == "discard":
                is_unique = False
                print("# Not unique.", i, candidate_ids[i], "is homolog to", accepted_ids[j], homology_score)
                break
                        
        if is_unique:
            accepted_sequences.append(candidate_sequences[i])
            accepted_ids.append(candidate_ids[i])
            accepted_topologies.append(candidate_topologies[i])
            print("# Unique.", i, len(accepted_sequences)-1, candidate_ids[i], homology_score)

    print("Accepted sequences:", len(accepted_ids))

    # Save filtered sequences
    output_path = "../data/" + seq_type + "_unique.3line"
    with open(output_path, "w") as f:
        for pid, seq, top in zip(accepted_ids, accepted_sequences, accepted_topologies):
            f.write(f">{pid}\n{seq}\n{top}\n")
    print(f"# Saved {len(accepted_ids)} sequences to {output_path}")

    

# MAIN   
def main():
    parser = ArgumentParser(description="Run Hobohm1 sequence filtering.")
    parser.add_argument("-sq", "--seq_type", required=True, help="Sequence file to process")
    args = parser.parse_args()
    run_hobohom1(args.seq_type)

if __name__ == "__main__":
    main()






