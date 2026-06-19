#!/usr/bin/env python
# coding: utf-8

# # Splitting the dataset 

# ## Python Imports

# In[1]:


import numpy as np
import random

#%matplotlib inline


# In[ ]:


from argparse import ArgumentParser

parser = ArgumentParser(description="SMM with Monte Carlo Method")

parser.add_argument("-sp+tm", action="store", dest="SP_TM_FILE", type=str, help="SP + TM peptides")
parser.add_argument("-tm", action="store", dest="TM_FILE", type=str, help="TM peptides")
parser.add_argument("-signal", action="store", dest="SIGNALS_FILE", type=str, help="Signals peptides")
parser.add_argument("-globular", action="store", dest="GLOBULAR_FILE", type=str, help="Globular peptides")
parser.add_argument("-train", action="store", dest="TRAIN_PROPORTION", type=float, default=0.8, help="Proportion of the dataset to use to form the training dataset (default 80%)")
#parser.add_argumennt("-test", action="store", dest="TEST_PROPORTION", type=float, default=0.2, help="Proportion of the dataset to use to form the test dataset (default 20%")
parser.add_argument("-downsize", action="store_true", dest="DOWNSIZE", help="Specify is downsizing is needed or not")


args = parser.parse_args()

sp_tm_file = args.SP_TM_FILE
tm_file = args.TM_FILE
signal_file = args.SIGNALS_FILE
glob_file = args.GLOBULAR_FILE
train_prop = args.TRAIN_PROPORTION
downsize = args.DOWNSIZE
#test_prop = args.TEST_PROPORTION #isn't needed bc equal 1-train_prop

#For each file, divide into a train and a test data set, add the flag (GLOBULAR / SIGNAL / SP+TM / TM) and then fuse in one file)

files_dict = {
    "sp_tm": sp_tm_file,
    "tm": tm_file,
    "signal": signal_file,
    "glob": glob_file
}

train_lines = []
test_lines = []

    
for prefix, f_path in files_dict.items():
    
    with open(f_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    
    peptides = []
    
    for i in range(0, len(lines), 3):
        
        if i + 2 < len(lines):
            header = lines[i]
            aa_seq = lines[i+1]
            state_seq = lines[i+2]
            
            header = header + "|" + prefix
            
            peptides.append((header, aa_seq, state_seq))
            
    if downsize and prefix in ["signal", "glob"]:
        random.shuffle(peptides) 
        peptides = peptides[:200]
            
    N_pept = len(peptides)
    N_train = int(train_prop * N_pept)
    
    random.shuffle(peptides)
    
    train_peptides = peptides[:N_train]
    test_peptides = peptides[N_train:]
    
    for pept in train_peptides:
        train_lines.extend(pept)  
        
    for pept in test_peptides:
        test_lines.extend(pept)


test_prop = round(1.0 - train_prop, 2)

train_output_file = f"train_dataset_fused_{train_prop*100}%.txt"
test_output_file = f"test_dataset_fused_{test_prop*100}%.txt"

with open(train_output_file, 'w') as file:
    for line in train_lines:
        file.write(line + "\n")  

with open(test_output_file, 'w') as file:
    for line in test_lines:
        file.write(line + "\n")

# A nice terminal printout to confirm it worked
print(f"Success! Datasets fused and saved.")
print(f"Total training peptides: {len(train_lines) // 3}")
print(f"Total testing peptides: {len(test_lines) // 3}")