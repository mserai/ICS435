from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import random 

def main():
    B1, B2 = make_blobs(n_samples=100, centers=3, cluster_std=.1, random_state=2)
    #B1, B2 = datasets.make_circles(n_samples=1000, random_state=2)
    #B1, B2 = datasets.make_gaussian_quantiles(n_samples=1000,n_features=2,random_state=12,n_classes=6)
    #B1, B2 = datasets.make_moons(n_samples=100, random_state=12)
#print(B1)
    K=2
    #for i in range(0,20):
    softKMeans(B1, K)

def softKMeans(B1, K, beta=1):
    centers = initializeClusterCenters(B1, K)
    distanceCriterion = 1
    convergenceCriterion = 0.1
    prev_cost = 0
    cost = 0 
    merge_counter = 0
    split_counter = 0
    remove_counter = 0
    while True:
        probabilities = calculateAssignments(centers, B1, beta)
        centers = updateClusterCenters(B1, probabilities, K)
        i = 0
        while i < len(centers):
            j = 0
            while j < len(centers):
                distance = np.linalg.norm(centers[i]-centers[j])
                print ("Distance: ", distance)
                if distance <= distanceCriterion and i != j:
                    print ("Cluster Merged: ", centers[j])
                    centers[i] = [(centers[i][0] + centers[j][0]) / 2, (centers[i][1] + centers[j][1]) / 2]
                    centers = np.delete(centers, j, 0)
                    i = -1
                    r_counter = []
                    remove_counter +=1
                    r_counter.append(remove_counter)
                    break
                else:
                    j += 1
            i += 1
        K = len(centers)
        print("New K : ", K)
        probabilities = calculateAssignments(centers, B1, beta)
        centers = updateClusterCenters(B1, probabilities, K)
        m_counter = []
        merge_counter +=1
        m_counter.append(merge_counter)
        cost = convergence(B1, probabilities, centers, K)
        if np.abs(cost - prev_cost) < convergenceCriterion:
            print ("Convergence Criterion Met. K =",K)
            break
        prev_cost = cost
        K *= 2
        split_centers = np.zeros((K, len(B1[0])))
        for i, center in enumerate(centers):
            split_centers[i * 2] = center
            split_centers[(i * 2) + 1] = [center[0] + (random.randint(0, 100) * 0.01), center[1] + (random.randint(0, 100) * 0.01)]
            print("New Center :", split_centers)
            s_counter = []
            split_counter +=1
            s_counter.append(split_counter)
        centers = split_centers
    print("Final Centers Before Plot:", centers)
    print("Split Counter = ", s_counter)
    print("Merge Counter = ", m_counter)
    print("Remove Counter = ", r_counter)
#     metric = []
#     metric = [(K,len(B1),s_counter, m_counter, r_counter)]
#     df = pd.DataFrame(metric)
#     df.to_csv('3CenterBlobsCompare.csv', mode='a',index=False, header=False)
    for center in centers:
        plt.plot(center[0], center[1],"ro")
    plt.scatter(B1[:,0], B1[:,1])
    plt.show()
    
def initializeClusterCenters(B1, K):
    centers = np.zeros((K, len(B1[0])))
    indexs = []
    for k in range(K):
        i = np.random.choice(len(B1))
        while i in indexs:
            i = np.random.choice(len(B1))
        indexs.append(i)
        centers[k] = B1[i]
    print("Initialized Centers: ", centers)
    return centers

def calculateAssignments(centers, B1, beta):
    K, D = centers.shape
    probabilities = np.zeros((len(B1), K))
    for i in range((len(B1))):        
        probabilities[i] = np.exp(-beta * np.linalg.norm(centers - B1[i], 2, axis=1))
    probabilities = probabilities/(probabilities.sum(axis=1, keepdims=True))
    print("Probabilities: ", probabilities)
    return probabilities

def updateClusterCenters(B1, probabilities, K):
    centers = np.zeros((K, len(B1[0])))
    for k in range(K):
        centers[k] = probabilities[:, k].dot(B1) / probabilities[:, k].sum()
    print("Updated Centers: ", centers)
    return centers

def convergence(B1, probabilities, centers, K):
    cost = 0
    for k in range(K):
        cost += (np.linalg.norm(B1 - centers[k], 2) * np.expand_dims(probabilities[:, k], axis=1)).sum()
    return cost
    
if __name__ == "__main__":
    main()
