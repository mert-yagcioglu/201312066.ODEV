#!/usr/bin/env python
# coding: utf-8

# In[21]:


#201312066
#Mert Yağcıoğlu
#19.12.2023 K-MEANS ve K-MEDOİDS ÖDEVİ

#NOT: Hocam MEDOİDS kütüphanesinin çalışması için önce #pip install pyclustering   kütüphanesinin yüklenmesi gerekli.






#1)K-MEANS İLE KÜMELEME
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



data = load_breast_cancer()
X = data.data
y = data.target

#Burda Normalizasyon yapıyoruz.
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)


#K-means
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)

#Model eğitiyoruz.
kmeans.fit(X_normalized)

#Küme merkezi ve etiketlerii
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

#PCA ile veriyi iki boyuta düşürme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

#Küme merkezlerini iki boyuta düşürdüğümüz için iki boyutta gösterme
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, color='red', label='Küme Merkezi')
plt.title('Meme Kanseri Verilerinde K-MEANS Kümeleme')
plt.xlabel('Ana Bileşen 1')
plt.ylabel('Ana Bileşen 2')
plt.legend()
plt.show()











#2)K-MEDOİDS İLE KÜMELEME
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.kmedoids import kmedoids                      #"pip install pyclustering"   kütüphanesinin yüklenmesi gerek
from pyclustering.utils import calculate_distance_matrix


data = load_breast_cancer()
X = data.data


#Burda Normalizasyon yapıyoruz tekrar
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

#K-means
k = 2             #Küme sayısı
initial_medoids = np.random.choice(range(len(X_normalized)), k, replace=False)
kmedoids_instance = kmedoids(calculate_distance_matrix(X_normalized), initial_medoids, data_type='distance_matrix')
kmedoids_instance.process()

#Kümeleme sonuçları
clusters = kmedoids_instance.get_clusters()

#Tüm küme merkezlerini belirliyoruz.
medoids = kmedoids_instance.get_medoids()


#PCA ile veriyi iki boyuta düşürüyoruz tekrar
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

#Küme merkezlerini ve kümeleme sonuçlarını görselleştiriyoruz.
plt.figure(figsize=(8, 6))
for cluster in clusters:
    plt.scatter(X_pca[cluster, 0], X_pca[cluster, 1], alpha=0.7)

#Bulunan Medoid'leri gösterme
plt.scatter(X_pca[medoids, 0], X_pca[medoids, 1], marker='X', s=200, color='red', label='Medoids')
plt.title('Meme Kanseri Verilerinde K-MEDOİDS Kümeleme')
plt.xlabel('Ana Bileşen 1')
plt.ylabel('Ana Bileşen 2')
plt.legend()
plt.show()



# In[ ]:





# In[ ]:




