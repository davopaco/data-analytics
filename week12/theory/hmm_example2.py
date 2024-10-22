import numpy as np

states = ["H1", "H2"]
observations = ["Heads", "Tails"]
obs_seq = [0, 1, 0]  # SECUENCIA DE OBSERVACIÓN: Heads (0), Tails (1), Heads (0)

start_prob = [0.5, 0.5] # PROBABILIDADES INICIALES
trans_prob = [[0.7, 0.3], [0.4, 0.6]]  # PROBABILIDADES DE TRANSICIÓN
emission_prob = [[0.9, 0.1], [0.2, 0.8]]  # PROBABILIDADES DE EMISIÓN

# ALGORITMO FORWARD
def forward(obs_seq, states, start_prob, trans_prob, emission_prob):
    n_states = len(states)
    n_observations = len(obs_seq)
    
    # INICIALIZAR LA MATRIZ
    fwd = np.zeros((n_states, n_observations))
    
    # PASO DE INICIALIZACIÓN
    for i in range(n_states):
        fwd[i, 0] = start_prob[i] * emission_prob[i][obs_seq[0]]
    
    # PASO RECURSIVO
    for t in range(1, n_observations):
        for j in range(n_states):
            fwd[j, t] = sum(fwd[i, t-1] * trans_prob[i][j] for i in range(n_states)) * emission_prob[j][obs_seq[t]]
    
    # PASO FINAL: SUMAR LAS PROBABILIDADES DE TODOS LOS ESTADOS EN LA ÚLTIMA OBSERVACIÓN
    final_prob = sum(fwd[i, n_observations-1] for i in range(n_states))
    
    return fwd, final_prob

# ALGORITMO BACKWARD
def backward(obs_seq, states, trans_prob, emission_prob):
    n_states = len(states)
    n_observations = len(obs_seq)
    
    # INICILIZAR LA MATRIZ
    bwd = np.zeros((n_states, n_observations))
    
    # PASO INICIAL: LLENAR LA ÚLTIMA COLUMNA CON UNOS
    for i in range(n_states):
        bwd[i, n_observations - 1] = 1.0
    
    # PASO RECURSIVO
    for t in range(n_observations - 2, -1, -1):
        for i in range(n_states):
            bwd[i, t] = sum(bwd[j, t+1] * trans_prob[i][j] * emission_prob[j][obs_seq[t+1]] for j in range(n_states))
    
    return bwd

# PROBABILIDADES POSTERIORES USANDO FORWARD-BACKWARD
def forward_backward(obs_seq, states, start_prob, trans_prob, emission_prob):
    fwd, final_prob = forward(obs_seq, states, start_prob, trans_prob, emission_prob) # PROBABILIDADES FORWARD Y TOTAL
    bwd = backward(obs_seq, states, trans_prob, emission_prob) # PROBABILIDADES BACKWARD
    
    n_states = len(states)
    n_observations = len(obs_seq)
    
    # INICIALIZAR LA MATRIZ
    posterior = np.zeros((n_states, n_observations))
    
    # CALCULAR LAS PROBABILIDADES POSTERIORES
    for t in range(n_observations):
        for i in range(n_states):
            posterior[i, t] = (fwd[i, t] * bwd[i, t]) / final_prob
    
    return posterior

posterior_probs = forward_backward(obs_seq, states, start_prob, trans_prob, emission_prob)

print(f"Secuencia de observaciones: {['Heads', 'Tails', 'Heads']}")
print("Probabilidades posteriores para cada paso:")
for t in range(len(obs_seq)):
    print(f"Observación {t} {observations[obs_seq[t]]}: H1: {posterior_probs[0, t]:.4f}, H2: {posterior_probs[1, t]:.4f}")
