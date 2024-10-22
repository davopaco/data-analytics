import numpy as np

# ESTADOS Y OBSERVACIONES
states = ["Sunny", "Cloudy", "Rainy"]
observations = ["Walk", "Shop", "Clean"]

start_prob = [0.3, 0.4, 0.3]  # PROBABILIDADES INICIALES DE LOS ESTADOS
trans_prob = [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.3, 0.0, 0.7]]  # PROBABILIDADES DE TRANSICIÓN
emission_prob = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3] ,[0.1, 0.4, 0.5]]  # PROBABILIDADES DE EMISIÓN

# SECUENCIA: Clean, Walk, Shop
observed_sequence = [2, 0, 1]

# ALGORITMO DE VITERBI
def viterbi(observed_sequence, states, start_prob, trans_prob, emission_prob):
    n_states = len(states)
    n_observations = len(observed_sequence)
    
    # INICIALIZAR TABLAS
    T1 = np.zeros((n_states, n_observations))  # TABLA DE PROBABILIDADES
    T2 = np.zeros((n_states, n_observations), dtype=int)  # TABLA DE CAMINOS

    # PASO INICIAL: Calcular la probabilidad inicial de cada estado.
    for i in range(n_states):
        T1[i, 0] = start_prob[i] * emission_prob[i][observed_sequence[0]]
        T2[i, 0] = -1

    # PASO RECURSIVO: PARA CADA OBSERVACIÓN, CALCULAR LA PROBABILIDAD MÁXIMA Y EL ESTADO QUE LA GENERÓ
    for t in range(1, n_observations):
        for j in range(n_states):
            max_prob = -1
            max_state = -1
            for i in range(n_states):
                prob = T1[i, t - 1] * trans_prob[i][j] * emission_prob[j][observed_sequence[t]]
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            T1[j, t] = max_prob
            T2[j, t] = max_state

    # PASO FINAL: ENCONTRAR EL ESTADO MÁS PROBABLE
    best_last_state = np.argmax(T1[:, n_observations - 1])
    best_path = [best_last_state]

    # TRACEBACK DE LOS ESTADOS MÁS PROBABLES
    for t in range(n_observations - 1, 0, -1):
        best_path.insert(0, T2[best_last_state, t])
        best_last_state = T2[best_last_state, t]

    return best_path, T1, T2

hidden_states_seq, T1, T2 = viterbi(observed_sequence, states, start_prob, trans_prob, emission_prob)

predicted_states = [states[i] for i in hidden_states_seq]
print("TABLA DE PROBABILIDADES (T1):")
print(T1)
print("\nTABLA DE CAMINOS (T2):")
print(T2)
print(f"SECUENCIA DE OBSERVACIONES: {list(map(lambda x: observations[x], observed_sequence))}")
print(f"HIDDEN STATES PREDICHOS: {predicted_states}")
