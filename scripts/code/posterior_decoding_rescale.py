#!/usr/bin/env python
# coding: utf-8

# # Posterior Decoding
# Implementation of the Posterior Decoding algorithm.
# 
# 

# ## Python Imports

# In[1]:


import numpy as np
import ast


# In[ ]:


from argparse import ArgumentParser
parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-i", action="store", dest="INPUT_SEQUENCE", type=str, help="Input file")
parser.add_argument("-ns", action="store", dest="NUMBER_STATES", type=str, help="Number of states")
parser.add_argument("-t", action="store", dest="TRANSITION_MATRIX", help="Transition matrix")
parser.add_argument("-e", action="store", dest="EMISSION_PROBS", help="Emission probabilities")
parser.add_argument("-p", action="store", dest="INITIAL_PROBS", help="Initial probabilities")
args = parser.parse_args()

input_sequence = args.INPUT_SEQUENCE
states = int(args.NUMBER_STATES) # Cast en entier

transition_matrix = np.array(ast.literal_eval(args.TRANSITION_MATRIX))
emission_probs = np.array(ast.literal_eval(args.EMISSION_PROBS))
initial_prob = np.array(ast.literal_eval(args.INITIAL_PROBS))


# ## Encode sequence as integers (index values)

# In[4]:


def encode( sequence, symbols):
    
    enc = [0] * len(sequence)
    
    for i in range(len(sequence)):
        enc[i] = symbols.find(sequence[i])
    
    return(enc)


# ## Parameters

# In[5]:


symbols = "ACDEFGHIKLMNPQRSTVWY"

input_encode = encode(input_sequence, symbols) 


# ## Forward Loop
# ### Remember that we here do NOT work in log space

# In[2]:


def initialize_forward(input_encode, states, initial_prob, emission_probs):
    
    alpha = np.zeros(shape=(states, len(input_encode)))
    G = 0
        
    for i in range(0, states): 
        
        alpha[i][0] = initial_prob[i]*emission_probs[i][input_encode[0]]
        G += alpha[i][0]        
        
    for i in range(0, states):
        alpha[i][0] /= G
        
    return alpha, G # On retourne G pour l'enregistrer


# In[6]:


alpha, first_G = initialize_forward(input_encode, states, initial_prob, emission_probs)

G_array = np.zeros(len(input_encode))
G_array[0] = first_G

# main loop

for i in range(1, len(input_encode)):
    G = 0
    
    for j in range(0, states):

        _sum = 0
        
        for k in range(0, states):
            
            _sum += alpha[k][i-1]*transition_matrix[k][j]            
         
        # store prob
        alpha[j][i] = emission_probs[j][input_encode[i]]*_sum
        G += alpha[j][i]

    for j in range(0, states):
        alpha[j][i] /= G
        
    G_array[i] = G # Enregistrement pour Baum-Welch


# ## Backward Loop

# In[7]:


def initialize_backward(input_encode, states):
    
    #beta = np.zeros(shape=(states, len(input_encode), dtype=float))
    beta = np.zeros(shape=(states, len(input_encode)))
        
    for i in range(0, states):
  
        beta[i][-1] = 1
        
    return beta


# In[8]:


beta = initialize_backward(input_encode, states)

# main loop
for i in range(len(input_encode)-2, -1, -1):
    
    G_beta = 0
    
    for j in range(0, states):

        _sum = 0

        for k in range(0, states):
            
            _sum += emission_probs[k][input_encode[i+1]] * transition_matrix[j][k] * beta[k][i+1]
        
        # store prob
        beta[j][i] = _sum
        G_beta += beta[j][i]

    for j in range(0, states):
        beta[j][i] /= G_beta


# ## Posterior Loop
#  

# In[11]:


# posterior (gamma)
gamma = np.zeros(shape=(states, len(input_encode)), dtype=float)

for i in range(0, len(input_encode)):
        
    # Calculate the normalization factor for this specific time step
    norm_factor = 0
    for j in range(0, states):
        norm_factor += alpha[j][i] * beta[j][i]
        
    for j in range(0, states):
        gamma[j][i] = (alpha[j][i] * beta[j][i]) / norm_factor 

# ## Xi Calculation

xi = np.zeros(shape=(states, states, len(input_encode)-1), dtype=float)

for t in range(0, len(input_encode)-1):
    norm_xi = 0
    for j in range(0, states):
        for k in range(0, states):
            xi[j][k][t] = alpha[j][t] * transition_matrix[j][k] * emission_probs[k][input_encode[t+1]] * beta[k][t+1]
            norm_xi += xi[j][k][t]
            
    for j in range(0, states):
        for k in range(0, states):
            xi[j][k][t] /= norm_xi




