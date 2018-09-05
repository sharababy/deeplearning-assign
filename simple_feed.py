import numpy as np 


x = np.asarray([1.5,2.5,3])

w1 = np.asarray([[0.05,0.05,0.05,0.05],
				[0.05,0.05,0.05,0.05],
				[0.05,0.05,0.05,0.05]])

w2 = np.asarray([[0.025,0.025,0.025],
				[0.025,0.025,0.025],
				[0.025,0.025,0.025],
				[0.025,0.025,0.025]])

w3 = np.asarray([[0.025,0.025,0.025,0.025,0.025],
				[0.025,0.025,0.025,0.025,0.025],
				[0.025,0.025,0.025,0.025,0.025]])


b1 = np.asarray([0.1,0.2,0.3,0.4])
b2 = np.asarray([5.2,3.2,4.3])
b3 = np.asarray([0.2,0.45,0.75,0.55,0.95])


a1 = x.dot(w1) + b1
a1 = 1.0 / (1 + np.exp(-a1))

a2 = a1.dot(w2) + b2
a2 = 1.0 / (1 + np.exp(-a2))

a3 = a2.dot(w3) + b3

a3 = np.exp(a3)
a3 = a3/np.sum(a3)

print(a3)
