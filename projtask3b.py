import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
np.random.seed(sum([ord(c) for c  in 'abiseria']))
img = cv2.imread('baboon.jpg')
h,w,d = img.shape
tx =0
zx=0
names = ['task3_baboon_3.jpg']
iter = [3]
for zx in range(len(iter)):
	img = cv2.imread('baboon.jpg')
	h,w,d = img.shape
	def calculatemean(cluster,Mu):
		meanx =0 
		meany = 0
	
		for j in range(iter[zx]):
			meanx =0 
			meany = 0
			meanz = 0
			for i in range(len(cluster[j])):
				meanx = meanx + cluster[j][i][0]
				meany = meany + cluster[j][i][1]
				meanz = meanz + cluster[j][i][2]
			meanx =  meanx + Mu[j][0]
			avgx = meanx/(len(cluster[j])+1)
			avgx = round(avgx,4)	
#print(avgx)
	
	
			meany =  meany + Mu[j][1]
			avgy = meany/(len(cluster[j])+1)
			avgy = round(avgy,4)	
#print(avgy)
			meanz =  meanz + Mu[j][2]
			avgz = meanz/(len(cluster[j])+1)
			avgz = round(avgz,4)
		
			Mu[j] = (avgx,avgy,avgz)

		#print(' new MU', Mu)
		return Mu
	
	def clusterization(Mu):
		min =0
		cluster = []
		for i in range(iter[zx]):
			cluster.append([])


		for i in range(h*w):
			dist2 =100
			for j in range(iter[zx]):
				a = np.array(X[i])
				b = np.array(Mu[j])
				dist = np.linalg.norm(a - b)
				if(dist < dist2):
					dist2 = dist
					min = j
			
			cluster[min].append(X[i])
		
		return cluster


#k = img[0]
#print(k)
	img = img.reshape(w*h,d)
	k= img.shape
	print(k)
	s =[]
	for i in range(iter[zx]):

		m = random.randint(0,k[0] )
		s.append(m)
#print( s)

	X = img
#print(len(X))

	Mu = []
	for i in range(iter[zx]):
		Mu.append(img[s[i]])
	
	for i in range(10):
		cluster = clusterization(Mu)
	#print (cluster)
		n = len(cluster[0])
	#print (n)
	
		q = calculatemean(cluster,Mu)
	#print (cluster)
		n = len(cluster[0])
	#print (n)
		Mu = q 
	
	print(Mu)
	
	'''for j in range(iter[zx]):
		n = len(cluster[j])
		for i in range(n):
			k = cluster[j][i][0]
			k = Mu[j][0]
			t = cluster[j][i][1]
			t = Mu[j][1]
			q = cluster[j][i][2]
			q= Mu[j][2]'''

	img = img.reshape(h,w,d)		
#cv2.imshow('img',img)
#cv2.waitKey(0)

	
	for i in range(512):
		dist2 =1000000000000000000000
		for j in range(512):
			dist2 =1000000000000000000000
			for s in range(iter[zx]):
		 		a = np.array(img[i][j])
		 		b = np.array(Mu[s])
		 		dist = np.linalg.norm(a - b)
		 		if(dist < dist2):
			 		dist2 = dist
			 		min = s
			img[i][j] = Mu[min]
			 

	cv2.imwrite(names[tx],img)
	
	tx = tx +1
'''
'task3_baboon_3.jpg','task3_baboon_5.jpg','task3_baboon_10.jpg','''
