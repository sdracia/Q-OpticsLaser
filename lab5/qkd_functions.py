def dataset_loading(file_path, t):
  state_mapping = {
    '00': 'H',
    '01': 'V',
    '10': 'D',
    '11': 'A'
  }

  decoy_mapping = {
    '00': 'N',
    '01': 'L',
    '10': 'H', 
    '11': 'X'
  }
  
  if t == "state":
    mapping = state_mapping
  elif t == "decoy":
    mapping = decoy_mapping
  else:
    raise ValueError(f"Invalid type: '{t}'. Accepted types are 'state' or 'decoy'.")
  
  with open(file_path, "rb") as file:
    states = []
  
    while byte := file.read(1):  # Read one byte at a time
      # Convert the byte to its binary representation
      binary_representation = bin(ord(byte))[2:].zfill(8)
      state_bits = binary_representation[-2:]

      # Map the bits to their corresponding state
      state = mapping.get(state_bits, "Unknown")

      states.append(state)
      
  return states


def divide_states(states, values):
  result = {}
  
  for value in values:
    result[value] = [state for state in states if state == value]
  return result


def cond_prob(transmitted_states, received_states, a, b):
  rec_a = [received_states[i] for i in range(len(received_states)) if transmitted_states[i] == a]
  rec_b_a = [state for state in rec_a if state == b]
  
  conditional_probability = len(rec_b_a) / len(rec_a)
  
  return conditional_probability


def total_cond_probs(transmitted_states, recieved_states, a_values, b_values):
  cond_probs = {}

  for a in a_values:
    for b in b_values:
      cond_probs[(b, a)] = cond_prob(transmitted_states, recieved_states, a, b)
      
  return cond_probs


def getQBER(transmitted_states, recieved_states, a_values, b_values):
  cond_probs = {}
  for a in a_values:
    for b in b_values:
      cond_probs[(b, a)] = cond_prob(transmitted_states, recieved_states, a, b)
      
  qber = sum(cond_probs.values())
      
  return cond_probs, qber