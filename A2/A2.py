import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

dataset = pd.read_csv('3D_spatial_network.txt', header=None)
X = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 3].values

X[:, 0] = (X[:, 0]-np.mean(X[:, 0]))/(np.std(X[:, 0]))
X[:, 1] = (X[:, 1]-np.mean(X[:, 1]))/np.std(X[:, 1])
y = (y-np.mean(y))/np.std(y)


def prepare(degree):
	powers = []
	for i in range(degree, -1, -1):
		for j in range(degree, -1, -1):
			if i + j <= degree:
				powers.append((i, j))
	z = (X[:, 0] ** 0) * (X[:, 1] ** 0)
	v = np.array(z)
	powers = powers[:-1]
	for i in powers:
		z = (X[:, 0] ** i[0]) * (X[:, 1] ** i[1])
		v = np.c_[v, z]
	# print(powers)
	return v, powers


def gradient_descent(iters, learning_rate):
	epoch_loss = []
	w = np.zeros(X.shape[1])
	for i in range(iters):
		preds = X.dot(w)
		# print('preds:==========\n', preds)
		error = preds - y
		loss = np.sum(error ** 2)
		epoch_loss.append(loss/len(X))
		# print(X.shape, error.shape)
		gradient = X.T.dot(error)/len(X)
		# print('gradient:==========\n', gradient)
		w -= learning_rate * gradient

	# print(epoch_loss)
	return w, epoch_loss


def gradient_descent_L1(iters, learning_rate, ld):
    epoch_loss = []
    w = np.zeros(X.shape[1])
    for i in range(iters):
        preds = X.dot(w)
        error = preds - y
        loss = np.sum(error ** 2)
        epoch_loss.append(loss/len(X))
        w2 = np.ones(w.shape[0])
        gradient = X.T.dot(error)/len(X) + (ld*w2)/2
        w = w - learning_rate*gradient
    return w, epoch_loss


def gradient_descent_L2(iters, learning_rate, ld):
	epoch_loss = []
	w = np.zeros(X.shape[1])
	for i in range(iters):
		preds = X.dot(w)
		# print('preds:==========\n', preds)
		error = preds - y
		loss = np.sum(error ** 2)
		epoch_loss.append(loss/len(X))
		# print(X.shape, error.shape)
		gradient = X.T.dot(error)/len(X) + w*ld
		# print('gradient:==========\n', gradient)
		w -= learning_rate * gradient

	# print(epoch_loss)
	return w, epoch_loss

def RMSE(w, a, b):
	preds = a.dot(w)
	error = preds - b
	error = np.sum(error ** 2)
	error = error/len(a)
	error = np.sqrt(error)
	return error

def R2(w, a, b):
	Y_pred = a.dot(w)
	mean_y = np.mean(b)
	ss_tot = sum((b - mean_y) ** 2)
	ss_res = sum((b - Y_pred) ** 2)
	r2 = 1 - (ss_res / ss_tot)
	return r2


iters = 10
X, powers = prepare(2)
X_tot = X
Y_tot = y
X = X[:400000, :]
y = y[:400000]
Xtest = X_tot[400000:, :]
Ytest = Y_tot[400000:]
w, el = gradient_descent(iters, 0.0001)
print("RMSE value for Degree 2 is for training Data is")
print(RMSE(w, X, y))
print("R Sq value for Degree 2 is for training Data is")
print(R2(w, X, y))
print("RMSE value for Degree 2 is for test Data is")
print(RMSE(w, Xtest, Ytest))
print("R Sq value for Degree 2 is for test Data is")
print(R2(w, Xtest, Ytest))
fig = plt.figure()
plt.plot(np.arange(0, iters), el)
fig.suptitle("Training Loss for degree 2")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
plt.savefig("Training Loss for degree 2")
plt.close()

X = X_tot
y = Y_tot

iters = 10
X, powers = prepare(3)
X_tot = X
Y_tot = y
X = X[:400000, :]
y = y[:400000]
Xtest = X_tot[400000:, :]
Ytest = Y_tot[400000:]
w, el = gradient_descent(iters, 0.0001)
print("RMSE value for Degree 3 is for training Data is")
print(RMSE(w, X, y))
print("R Sq value for Degree 3 is for training Data is")
print(R2(w, X, y))
print("RMSE value for Degree 3 is for test Data is")
print(RMSE(w, Xtest, Ytest))
print("R Sq value for Degree 3 is for test Data is")
print(R2(w, Xtest, Ytest))
fig = plt.figure()
plt.plot(np.arange(0, iters), el)
fig.suptitle("Training Loss for degree 3")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
plt.savefig("Training Loss for degree 3")
plt.close()

X = X_tot
y = Y_tot

iters = 10
X, powers = prepare(4)
X_tot = X
Y_tot = y
X = X[:400000, :]
y = y[:400000]
Xtest = X_tot[400000:, :]
Ytest = Y_tot[400000:]
w, el = gradient_descent(iters, 0.0001)
print("RMSE value for Degree 4 is for training Data is")
print(RMSE(w, X, y))
print("R Sq value for Degree 4 is for training Data is")
print(R2(w, X, y))
print("RMSE value for Degree 4 is for test Data is")
print(RMSE(w, Xtest, Ytest))
print("R Sq value for Degree 4 is for test Data is")
print(R2(w, Xtest, Ytest))
fig = plt.figure()
plt.plot(np.arange(0, iters), el)
fig.suptitle("Training Loss for degree 4")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
plt.savefig("Training Loss for degree 4")
plt.close()

X = X_tot
y = Y_tot

iters = 10
X, powers = prepare(5)
X_tot = X
Y_tot = y
X = X[:400000, :]
y = y[:400000]
Xtest = X_tot[400000:, :]
Ytest = Y_tot[400000:]
w, el = gradient_descent(iters, 0.0001)
print("RMSE value for Degree 5 is for training Data is")
print(RMSE(w, X, y))
print("R Sq value for Degree 5 is for training Data is")
print(R2(w, X, y))
print("RMSE value for Degree 5 is for test Data is")
print(RMSE(w, Xtest, Ytest))
print("R Sq value for Degree 5 is for test Data is")
print(R2(w, Xtest, Ytest))
fig = plt.figure()
plt.plot(np.arange(0, iters), el)
fig.suptitle("Training Loss for degree 5")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
plt.savefig("Training Loss for degree 5")
plt.close()

X = X_tot
y = Y_tot

iters = 10
X, powers = prepare(6)
X_tot = X
Y_tot = y
X = X[:400000, :]
y = y[:400000]
Xtest = X_tot[400000:, :]
Ytest = Y_tot[400000:]
w, el = gradient_descent(iters, 0.0001)
print("RMSE value for Degree 6 is for training Data is")
print(RMSE(w, X, y))
print("R Sq value for Degree 6 is for training Data is")
print(R2(w, X, y))
print("RMSE value for Degree 6 is for test Data is")
print(RMSE(w, Xtest, Ytest))
print("R Sq value for Degree 6 is for test Data is")
print(R2(w, Xtest, Ytest))
fig = plt.figure()
plt.plot(np.arange(0, iters), el)
fig.suptitle("Training Loss for degree 6")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
plt.savefig("Training Loss for degree 6")
plt.close()

X = X_tot
y = Y_tot

iters = 10
X, powers = prepare(6)
X_tot = X
Y_tot = y
X = X[:400000, :]
y = y[:400000]
Xtest = X_tot[400000:, :]
Ytest = Y_tot[400000:]
w, el = gradient_descent_L1(iters, 0.0001, 1)
print("RMSE value for Degree 6 is for training Data is")
print(RMSE(w, X, y))
print("R Sq value for Degree 6 is for training Data is")
print(R2(w, X, y))
print("RMSE value for Degree 6 is for test Data is")
print(RMSE(w, Xtest, Ytest))
print("R Sq value for Degree 6 is for test Data is")
print(R2(w, Xtest, Ytest))
fig = plt.figure()
plt.plot(np.arange(0, iters), el)
fig.suptitle("Training Loss for degree 6")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
plt.savefig("Training Loss for degree 6")
plt.close()

X = X_tot
y = Y_tot

iters = 10
X, powers = prepare(6)
X_tot = X
Y_tot = y
X = X[:400000, :]
y = y[:400000]
Xtest = X_tot[400000:, :]
Ytest = Y_tot[400000:]
w, el = gradient_descent_L2(iters, 0.0001, 1)
print("RMSE value for Degree 6 is for training Data is")
print(RMSE(w, X, y))
print("R Sq value for Degree 6 is for training Data is")
print(R2(w, X, y))
print("RMSE value for Degree 6 is for test Data is")
print(RMSE(w, Xtest, Ytest))
print("R Sq value for Degree 6 is for test Data is")
print(R2(w, Xtest, Ytest))
fig = plt.figure()
plt.plot(np.arange(0, iters), el)
fig.suptitle("Training Loss for degree 6")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
plt.savefig("Training Loss for degree 6")
plt.close()