import numpy as np




def distance_matrix(S):
	if S.ndim == 2:
		D = np.empty((S.shape[0],S.shape[0]))
		for i in np.arange(S.shape[0]):
			for j in np.arange(S.shape[0]):
				D[i,j]=np.linalg.norm((S[i,:]-S[j,:]), ord=2, axis=0)

	else: 
		D = np.empty((S.shape[0],S.shape[0]))
		for i in np.arange(S.shape[0]):
			for j in np.arange(S.shape[0]):
				D[i,j]=np.linalg.norm((S[i]-S[j]))

	return D


def remove_redundant(S, epsilon):
	D = distance_matrix(S)
	indices = []
	for i in np.arange(D.shape[0]):
		for j in np.arange(i+1,D.shape[1]):
			if D[i,j] < epsilon:
				indices.append(j)
	if S.ndim == 1:
		S_new = np.delete(S, indices)
	else:
		S_new = np.delete(S, indices, axis=0)
	return S_new


S = np.array([[0,0],[0.01,0],[0.02,0],[0.1,0]])

print(S.shape)
S_new = remove_redundant(S, .03)

print(S_new)
