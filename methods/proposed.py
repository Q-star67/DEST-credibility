import numpy as np
import math
import itertools
from itertools import combinations
from math import factorial


m1=np.array([0.5054,0.2591,0.1766,0.0589])
m2=np.array([0.0000,0.7386,0.1501,0.1113])
m3=np.array([0.5676,0.1909,0.0000,0.2416])
m4=np.array([0.4787,0.2878,0.2334,0.0000])
m5=np.array([0.6085,0.1216,0.0000,0.2699])
evidence_matrix= np.array([m1, m2, m3,m4,m5])


"1.Jousselme distance"
m1 = {'A': 0.5054, 'B': 0.2591, 'C': 0.1766, 'D': 0.0589}
m2 = {'A': 0.0000, 'B': 0.7386, 'C': 0.1501, 'D': 0.1113}
m3 = {'A': 0.5676, 'B': 0.1909, 'C': 0.0000, 'D': 0.2416}
m4 = {'A': 0.4787, 'B': 0.2878, 'C': 0.2334, 'D': 0.0000}
m5 = {'A': 0.6085, 'B': 0.1216, 'C': 0.0000, 'D': 0.2699}


def jousselme_distance(m1, m2, frame_of_discernment):
    # Define the subsets in the frame of discernment
    subsets = list(frame_of_discernment.keys())

    # Convert the BPAs to vectors
    m1_vector = np.array([m1.get(subset, 0) for subset in subsets])
    m2_vector = np.array([m2.get(subset, 0) for subset in subsets])

    # D matrix
    D = np.zeros((len(subsets), len(subsets)))
    for i, A_i in enumerate(subsets):
        for j, A_j in enumerate(subsets):
            intersection = set(frame_of_discernment[A_i]) & set(frame_of_discernment[A_j])
            union = set(frame_of_discernment[A_i]) | set(frame_of_discernment[A_j])
            D[i, j] = len(intersection) / len(union)

    # Calculate the Jousselme distance
    diff = m1_vector - m2_vector
    distance = np.sqrt(0.5 * np.dot(np.dot(diff, D), diff.T))

    return distance

# Define the frame of discernment
frame_of_discernment = {
    'A': ['A'],
    'B': ['B'],
    'C': ['C'],
    'D': ['D'],
}
evidences = [m1, m2, m3, m4, m5]

# Number of pieces of evidence
n = len(evidences)

# Initialize the Jousselme distance matrix
jousselme_distance_matrix = np.zeros((n, n))

# Calculate the Jousselme distance matrix
for i in range(n):
    for j in range(n):
        jousselme_distance_matrix[i, j] = jousselme_distance(evidences[i], evidences[j], frame_of_discernment)

# Calculate the average distance matrix
average_distance_matrix = np.zeros(n)
for i in range(n):
    sum_distances = np.sum(jousselme_distance_matrix[i]) - jousselme_distance_matrix[i, i] # Exclude diagonal element
    average_distance_matrix[i] = sum_distances / (n - 1)


similarity_matrix = 1 - jousselme_distance_matrix


#Print the matrices
print("jousselme_distance_matrix:\n", jousselme_distance_matrix)
print("similarity_matrix:\n", similarity_matrix)



"2.Shapely"
def non_linear_gain(similarity, alpha=10, beta=0.55, s=1, c=-0.5):
    return s * (1 / (1 + np.exp(-alpha * (similarity - beta)))) + c

def calculate_pairwise_payoff(i, j, similarity_matrix, initial_payoffs):
    sim = similarity_matrix[i, j]
    gain = non_linear_gain(sim)
    total_payoff = initial_payoffs[i] + initial_payoffs[j]
    return total_payoff + total_payoff * (gain)

def calculate_triple_payoff(i, j, k, similarity_matrix, initial_payoffs):
    sim_ij = similarity_matrix[i, j]
    sim_ik = similarity_matrix[i, k]
    sim_jk = similarity_matrix[j, k]
    effective_similarity = (sim_ij + sim_ik + sim_jk) / 3
    gain = non_linear_gain(effective_similarity)
    total_payoff = initial_payoffs[i] + initial_payoffs[j] + initial_payoffs[k]
    return total_payoff + total_payoff * (gain)

def calculate_group_payoff(players, similarity_matrix, initial_payoffs):
    total_payoff = sum(initial_payoffs[player] for player in players)
    if len(players) < 2:
        return total_payoff

    total_sim = 0
    count = 0
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            total_sim += similarity_matrix[players[i], players[j]]
            count += 1
    avg_sim = total_sim / count
    gain = non_linear_gain(avg_sim)
    return total_payoff + total_payoff * (gain)

def shapley_value(similarity_matrix, initial_payoffs):
    num_players = len(initial_payoffs)
    shapley_values = np.zeros(num_players)
    players = list(range(num_players))

    for i in players:
        for S in combinations(players, r=len(players) - 1):
            if i in S:
                continue
            S_with_i = S + (i,)
            marginal_contribution = calculate_group_payoff(S_with_i, similarity_matrix, initial_payoffs) - calculate_group_payoff(S, similarity_matrix, initial_payoffs)
            shapley_values[i] += marginal_contribution / (factorial(len(S)) * factorial(num_players - len(S) - 1))

        shapley_values[i] /= factorial(num_players)

    return shapley_values

def generate_combinations_and_payoffs(similarity_matrix, initial_payoffs):
    num_evidences = len(initial_payoffs)

    shapley_vals = shapley_value(similarity_matrix, initial_payoffs)
    shapley_vals_normalized = shapley_vals / np.sum(shapley_vals)

    result = {
        "shapley_values": shapley_vals_normalized,
        "combinations": []
    }

    for r in range(2, num_evidences + 1):
        for combo in combinations(range(num_evidences), r):
            if r == 2:
                payoff = calculate_pairwise_payoff(combo[0], combo[1], similarity_matrix, initial_payoffs)
            elif r == 3:
                payoff = calculate_triple_payoff(combo[0], combo[1], combo[2], similarity_matrix, initial_payoffs)
            else:
                payoff = calculate_group_payoff(combo, similarity_matrix, initial_payoffs)
            result["combinations"].append((combo, payoff))

    return result

initial_payoffs = [1, 1, 1, 1, 1]
result = generate_combinations_and_payoffs(similarity_matrix, initial_payoffs)
shap = np.array(result["shapley_values"])
print("shap：\n", shap)


"3."

def calculate_deng_entropy(m):
    deng_entropy = 0.0
    for belief in m:
        hypothesis, probability = belief.split(':')
        probability = float(probability)
        if probability > 0:
            cardinality = 1 if len(hypothesis) == 1 else 2
            deng_entropy -= probability * np.log2(probability / (2 ** cardinality - 1))
    return deng_entropy



m1 = ["A:0.5054", "B:0.2591", "C:0.1766","D:0.0589"]
m2 = ["A:0.0000", "B:0.7386", "C:0.1501","D:0.1113"]
m3 = ["A:0.5676", "B:0.1909", "C:0.0000","D:0.2416"]
m4 = ["A:0.4787", "B:0.2878", "C:0.2334","D:0.0000"]
m5 = ["A:0.6085", "B:0.1216", "C:0.0000","D:0.2699"]


Ed = np.zeros(5)
Ed[0] = calculate_deng_entropy(m1)
Ed[1] = calculate_deng_entropy(m2)
Ed[2] = calculate_deng_entropy(m3)
Ed[3] = calculate_deng_entropy(m4)
Ed[4] = calculate_deng_entropy(m5)

IV=np.exp(Ed)
IV_=IV/np.sum(IV)
print("IV_:")
print(IV_)


# source_credibility = np.array([])
# AIV = source_credibility * IV_
# AIV_ = AIV / np.sum(AIV)
# print("AIV_:", AIV_)


ACrd=IV_*shap
print("ACrd:",ACrd)
#ACrd_
ACrd_=ACrd/np.sum(ACrd)
ACrd_=np.array([ACrd_])
print("ACrd_: ",ACrd_)


combined_evidence=np.dot(ACrd_,evidence_matrix)
print("combined_evidence：",combined_evidence)

evidences = {
    "A":0.49890781 ,
    "B":0.25718509 ,
    "C":0.11962217 ,
    "D":0.12428276
}


def combine_evidence(m1, m2):
    combined = {}
    conflict = 0

    for key1, value1 in m1.items():
        for key2, value2 in m2.items():
            intersection = ''.join(sorted(set(key1).intersection(set(key2))))
            if intersection:
                if intersection in combined:
                    combined[intersection] += value1 * value2
                else:
                    combined[intersection] = value1 * value2
            else:
                conflict += value1 * value2

    if conflict == 1:
        raise ValueError("Total conflict, no combination possible.")

    normalization_factor = 1 / (1 - conflict)
    for key in combined:
        combined[key] *= normalization_factor

    return combined

#N-1：
evidence1 = evidences
for _ in range(4):
    evidence1 = combine_evidence(evidence1, evidences)

print(evidence1)





