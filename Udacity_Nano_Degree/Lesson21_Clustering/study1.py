# https://towardsdatascience.com/k-means-clustering-introduction-to-machine-learning-algorithms-c96bf0d5d57a
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


df = pd.read_csv('iris.csv')

classes = df['species']

df = df.drop(['species'], axis=1)

data = df.values.tolist()
data = np.array(data)

data, classes = shuffle(data, classes)

train_data = data[:135]
test_data = data[135:]
print("Train : {} and Test : {}".format(len(train_data), len(test_data)))
## Randomly place Centroids of the 3 clusters

train_data = data[:135]
# c1 = [6.0, 2.0, 5.0, 1.0]
c1 = [float(np.random.randint(4,8)),float(np.random.randint(1,5)),
      float(np.random.randint(1,7)),float(np.random.randint(0,3))]
c2 = [float(np.random.randint(4,8)),float(np.random.randint(1,5)),
      float(np.random.randint(1,7)),float(np.random.randint(0,3))]
c3 = [float(np.random.randint(4,8)),float(np.random.randint(1,5)),
      float(np.random.randint(1,7)),float(np.random.randint(0,3))]

pred_train = [0] * len(train_data)
epochs = 1
fig = plt.figure(figsize=(9,7))
fig.tight_layout()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

while( epochs <= 100):
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []

    for i, point in enumerate(train_data):
        # Find the eucledian distance between all points and the centroid
        dis_point_c1 = ( (c1[0] - point[0])**2 + (c1[1] - point[1])**2 +
                        (c1[2] - point[2])**2 + (c1[3] - point[3])**2)**0.5
        dis_point_c2 = ( (c2[0] - point[0])**2 + (c2[1] - point[1])**2 +
                        (c2[2] - point[2])**2 + (c2[3] - point[3])**2)**0.5
        dis_point_c3 = ( (c3[0] - point[0])**2 + (c3[1] - point[1])**2 +
                        (c3[2] - point[2])**2 + (c3[3] - point[3])**2)**0.5
        distances = [dis_point_c1,dis_point_c2,dis_point_c3]
        
        pos = distances.index(min(distances))
        pred_train[i] = pos
        if(pos == 0):
            cluster_1.append(point)
        elif(pos==1):
            cluster_2.append(point)
        else:
            cluster_3.append(point)
        
    prev_c1 = c1
    prev_c2 = c2
    prev_c3 = c3

    cluster_1 = np.array(cluster_1)
    cluster_2 = np.array(cluster_2)
    cluster_3 = np.array(cluster_3)

    # Find the mean of all points withing a cluster to get the new centroid
    if(len(cluster_1) != 0):
        c1 = [sum(cluster_1[:,0]) / float(len(cluster_1)),
              sum(cluster_1[:,1]) / float(len(cluster_1)),
              sum(cluster_1[:,2]) / float(len(cluster_1)),
              sum(cluster_1[:,3]) / float(len(cluster_1))]
    if(len(cluster_2) != 0):
        c2 = [sum(cluster_2[:,0]) / float(len(cluster_2)),
              sum(cluster_2[:,1]) / float(len(cluster_2)),
              sum(cluster_2[:,2]) / float(len(cluster_2)),
              sum(cluster_2[:,3]) / float(len(cluster_2))]
    if(len(cluster_3) != 0):
        c3 = [sum(cluster_3[:,0]) / float(len(cluster_3)),
              sum(cluster_3[:,1]) / float(len(cluster_3)),
              sum(cluster_3[:,2]) / float(len(cluster_3)),
              sum(cluster_3[:,3]) / float(len(cluster_3))]
       
    print(epochs)
    epochs +=1
    centers = np.column_stack((c1,c2,c3)).T
    ax1.cla()
    ax1.scatter(train_data[:,0],train_data[:,1], c=pred_train, s=50, cmap='viridis' )
    centers = np.column_stack((c1,c2,c3)).T
    ax1.scatter(centers[:,0], centers[:,1],c=[1,2,3], s=200, alpha=0.5, marker='o')
    ax1.set_title('Iteration {}'.format(epochs))
    plt.pause(1)
    
    if(prev_c1 == c1 and prev_c2 == c2 and prev_c3 == c3):
        print("Converged after {}".format(epochs))
        break 



pred = []

for point in test_data:
    dis_point_c1 = ( (c1[0] - point[0])**2 + (c1[1] - point[1])**2 +
                    (c1[2] - point[2])**2 + (c1[3] - point[3])**2)**0.5
    dis_point_c2 = ( (c2[0] - point[0])**2 + (c2[1] - point[1])**2 +
                    (c2[2] - point[2])**2 + (c2[3] - point[3])**2)**0.5
    dis_point_c3 = ( (c3[0] - point[0])**2 + (c3[1] - point[1])**2 +
                    (c3[2] - point[2])**2 + (c3[3] - point[3])**2)**0.5
    distances = [dis_point_c1,dis_point_c2,dis_point_c3]
    distances = [dis_point_c1,dis_point_c2,dis_point_c3]

    pos = distances.index(min(distances))
    pred.append(pos)
print('*'*70)
print(pred)
print('*'*70)

ax2.scatter(test_data[:,0],test_data[:,1], c=pred, s=50, cmap='viridis' )
centers = np.column_stack((c1,c2,c3)).T
ax2.scatter(centers[:,0], centers[:,1],c=[1,2,3], s=200, alpha=0.5)
plt.show()