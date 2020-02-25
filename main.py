'''chane the below arguments to check different tasks'''
TRAINSIZE = 5000
TESTSIZE = 500
'''To check TASK 3 put Normalize=1 otherwise 0'''
Nomalize = 1
learningRate = 0.01
threshold = 85

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


def normalize(data):
    for i in range(len(data)):
        data[i] = data[i] / 255
    return data


'''Chaneg the below numbers to pick how many samples you need'''
trainData = np.loadtxt("mnist_train.csv", delimiter=",", max_rows=TRAINSIZE)
testData = np.loadtxt("mnist_test.csv", delimiter=",", max_rows=TESTSIZE)
print(trainData.shape)
print(testData.shape)

# Step 0: Normalization to have 0 and 1

trainImg = np.asfarray(trainData[:, 1:])
testImg = np.asfarray(testData[:, 1:])

# to normalize dataset with binary function
if Nomalize == 0:
    trainImg[trainImg < threshold] = 0
    trainImg[trainImg >= threshold] = 1
    testImg[testImg < threshold] = 0
    testImg[testImg >= threshold] = 1
else:
    # to normalize dataset in range [0,1]
    trainImg = normalize(trainImg)
    testImg = normalize(testImg)

train_labels = np.asfarray(trainData[:, :1])
test_labels = np.asfarray(testData[:, :1])

no_of_different_labels = 10
lr = np.arange(10)
train_labels_one_hot = (lr == train_labels).astype(np.float)
test_labels_one_hot = (lr == test_labels).astype(np.float)

# Step 1: Initialize parameters and weights
inputNodes = 784
outputNodes = 10

epoch = 1
w = np.zeros((outputNodes, inputNodes + 1))
w[:, :] = 0.1

# Step 2: Apply input x from training set


MSE = []

while epoch < 50:

    mse = []
    for idx in range(len(trainImg)):
        x = trainImg[idx]

        d = train_labels_one_hot[idx]
        V = np.dot(w[:, 1:], x) + w[:, 0]
        Y = np.zeros(outputNodes)
        # step 4: applying activation function
        for i in range(outputNodes):
            if Nomalize == 0:
                if V[i] >= 0:
                    Y[i] = 1
                else:
                    Y[i] = 0
            else:
                Y[i] = sigmoid(V[i])
        e = d - Y

        # e= np.array([e])
        w[:, 1:] += (learningRate * (e[:,None] * x[None,:]))
        w[:, 0] += learningRate * e
        # print("MSE: ", float(MSE))
        mse.append(np.sum((d - Y) ** 2))
    MSE.append(np.sum(mse) / 2)
    epoch += 1
    if MSE[-1] < 0.001:
        break
    # print("epoch: ", epoch,", MSE:", MSE)

fig, ax = plt.subplots()
numberArrayTestIncorrect = np.zeros(10)
numberArrayTest = np.zeros(10)
ax.plot(MSE)
ax.set(xlabel='Iteration', ylabel='MSE', title='Learning curve for learning rate=' + str(learningRate))
ax.grid()
plt.show()

# testing process:
correct = []
incorrect = []
for idx in range(len(testImg)):
    x = testImg[idx][np.newaxis]
    x = x.T
    checkIdx = int(test_labels[idx][0])
    d = test_labels_one_hot[idx]
    V = np.dot(w[:, 1:], x) + w[:, 0][np.newaxis].T
    Y = np.zeros(outputNodes)
    for i in range(outputNodes):
        if V[i] >= 0:
            Y[i] = 1
        else:
            Y[i] = 0

    if np.array_equal(d, Y):
        correct.append(1)
        numberArrayTest[checkIdx] += 1
    else:
        incorrect.append(1)
        numberArrayTestIncorrect[checkIdx] += 1
print("Correct=", sum(correct), ", incorrect: ", sum(incorrect), ", accuracy: ", sum(correct)/TESTSIZE)
print(numberArrayTest)
print(numberArrayTestIncorrect)

N = 10
fig, ax = plt.subplots()
ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars: can also be len(x) sequence

p1 = ax.bar(ind, numberArrayTest, width, bottom=0, yerr=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
p2 = ax.bar(ind + width, numberArrayTestIncorrect, width, bottom=0, yerr=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

ax.set_title('Correct VS incorrect identification')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
ax.legend((p1[0], p2[0]), ('Correct', 'Incorrect'))
ax.autoscale_view()
plt.show()
