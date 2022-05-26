import numpy as np
from math import *
##### FISHERFACE #####

def normalize(X, low, high, dtype=None):
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	X = X - float(minX)
	X = X / float((maxX - minX))

	X = X * (high - low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

def asRowMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0,X[0].size), dtype=X[0].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
	return mat

def project(W, X, mu=None):
	if mu is None:
		return np.dot(X,W)
	return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
	if mu is None:
		return np.dot(Y, W.T)
	return np.dot(Y, W.T) + mu

def pca(X, y, num_components = 0):
	[n,d] =X.shape
	if (num_components <=0) or (num_components>n):
		num_components = n
	mu = X.mean(axis = 0)
	X = X - mu
	if n>d:
		C = np.dot(X.T, X)
		[eigenvalues, eigenvectors] =  np.linalg.eigh(C)
	else:
		C = np.dot(X, X.T)
		[eigenvalues, eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T, eigenvectors)
		for i in range(n):
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]

	eigenvalues = eigenvalues[0:num_components].copy()
	eigenvectors = eigenvectors[:,0:num_components].copy()
	return [eigenvalues, eigenvectors, mu]

def lda(X, y, num_components = 0):
	y = np.asarray(y)
	[n,d] = X.shape
	c = np.unique(y)
	if (num_components<=0) or (num_components>(len(c)-1)):
		num_components = (len(c)-1)
	meanTotal = X.mean(axis=0)
	Sw = np.zeros((d,d), dtype=np.float32)
	Sb = np.zeros((d,d), dtype=np.float32)
	for i in c:
		Xi = X[np.where(y==i)[0],:]
		meanClass = Xi.mean(axis=0)
		Sw = Sw + np.dot((Xi - meanClass).T,(Xi-meanClass))
		Sb = Sb + n * np.dot((meanClass - meanTotal).T,(meanClass - meanTotal))
	eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
	idx =np.argsort(-eigenvalues.real)
	eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
	eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
	eigenvectors = np.array(eigenvectors[0:,0:num_components].real, dtype=np.float32, copy=True)
	return [eigenvalues, eigenvectors]

def fisherfaces(X,y, num_components = 0):
	y = np.asarray(y)
	[n, d] = X.shape
	c = len(np.unique(y))
	[eigenvalue_pca, eigenvector_pca, mu_pca] = pca(X, y, (n-c))
	[eigenvalue_lda, eigenvector_lda] = lda(project(eigenvector_pca, X, mu_pca), y, num_components)
	eigenvector = np.dot(eigenvector_pca, eigenvector_lda)
	return [ eigenvalue_lda, eigenvector, mu_pca]


class EuclideanDistance():
	def __call__(self, p, q):
		p = np.asarray(p).flatten()
		q = np.asarray(q).flatten()
		return sqrt(sum(np.power((p-q),2)))

class ModelFisherfaces(object):
	def __init__(self,  X=None, y=None, dist_metric=EuclideanDistance(), num_components = 0):
		self.dis_metric = dist_metric
		self.num_components = 0
		self.projections = []
		self.mu = []
		self.W = []
		if (X is not None) and (y is not None):
			self.compute(X,y) 
	def compute(self, X, y):
		[D,self.W, self.mu] = fisherfaces(asRowMatrix(X), y, self.num_components)

		self.y = y
		for xi in X:
			self.projections.append(project(self.W, xi.reshape(1,-1),self.mu))

	def predict(self, X):
		minDist = np.finfo('float').max
		minClass = -1
		Q = project(self.W, X.reshape(1,-1), self.mu)
		for i in range(len(self.projections)):
			dist = self.dis_metric(self.projections[i], Q)
			if dist < minDist:
				minDist = dist
				minClass = self.y[i]
		return minClass

	def extract(self, X):
		return project(self.W, X.reshape(1,-1),self.mu)

# END FISHERFACE