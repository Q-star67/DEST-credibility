import numpy as np


m1=np.array([0.5054,0.2591,0.1766,0.0589])
m2=np.array([0.0000,0.7386,0.1501,0.1113])
m3=np.array([0.5676,0.1909,0.0000,0.2416])
m4=np.array([0.4787,0.2878,0.2334,0.0000])
m5=np.array([0.6085,0.1216,0.0000,0.2699])
evidence_matrix= np.array([m1, m2, m3,m4,m5])


"1.BJS"
def BJS_divergence(m1, m2):
    total_sum = 0.0
    epsilon = 1e-12
    for i in range(len(m1)):
        if m1[i] == 0 or m2[i] == 0 or m1[i] + m2[i] < epsilon:
            term1 = 0 if m1[i] == 0 else m1[i] * np.log2(2 * (m1[i] + epsilon) / (m1[i] + m2[i] + epsilon))
            term2 = 0 if m2[i] == 0 else m2[i] * np.log2(2 * (m2[i] + epsilon) / (m1[i] + m2[i] + epsilon))
        else:
            term1 = m1[i] * np.log2(2 * m1[i] / (m1[i] + m2[i]))
            term2 = m2[i] * np.log2(2 * m2[i] / (m1[i] + m2[i]))
        total_sum += 0.5 * (term1 + term2)
    return total_sum

def BJS_divergence_matrix(evidence_matrix):
    num_distributions = evidence_matrix.shape[0]
    bjs_matrix = np.zeros((num_distributions, num_distributions))

    # Calculate BJS divergence for each pair of distributions
    for i in range(num_distributions):
        for j in range(num_distributions):
            if i != j:
                bjs_matrix[i, j] = BJS_divergence(evidence_matrix[i], evidence_matrix[j])

    return bjs_matrix

bjs_matrix = BJS_divergence_matrix(evidence_matrix)
print("BJS_matrix:")
print(bjs_matrix)
average_bjs = np.sum(bjs_matrix, axis=1) / (bjs_matrix.shape[1] - 1)
print("average_bjs:")
print(average_bjs)


Sup=1/average_bjs
print("Sup_matrix:")
print(Sup)
Crd=Sup/np.sum(Sup)
print("Crd:")
print(Crd)

"2."
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


print("Ed:")
print(Ed)

IV=np.exp(Ed)
print("IV:")
print(IV)

IV_=IV/np.sum(IV)
print("IV_:")
print(IV_)

ACrd=IV_*Crd
print("ACrd:")
print(ACrd)

ACrd_=ACrd/np.sum(ACrd)
ACrd_=np.array([ACrd_])
print("ACrd_:")
print(ACrd_)



combined_evidence=np.dot(ACrd_,evidence_matrix)
print("combined_evidence：",combined_evidence)


evidences = {
    "A":0.49214327 ,
    "B":0.26421037 ,
    "C":0.12275075 ,
    "D":0.12089458
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