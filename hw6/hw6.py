import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import glob
import os
from mlxtend.data import loadlocal_mnist
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

# over-determined system
lam = 0.1 # set regularization parameter for sparsity promotion
alpha = [0.1, 0.2, 0.5, 0.7, 0.8, 1.0]

# reads in MNIST data images and labels
train_image, train_label = loadlocal_mnist(
        images_path= os.getcwd() + '/train_images/train-images-idx3-ubyte',
        labels_path= os.getcwd() + '/train_labels/train-labels-idx1-ubyte')

test_image, test_label = loadlocal_mnist(
        images_path= os.getcwd() + '/test_images/t10k-images.idx3-ubyte',
        labels_path= os.getcwd() + '/test_labels/t10k-labels-idx1-ubyte')

train_label_converted = np.zeros((train_label.shape[0], 10))

# converts labels to vectors
def convert_label(number):
    vector_label = np.zeros(10)
    vector_label[number-1] = 1
    return vector_label

# converts vectors to labels
def convert_back(vector):
    for i in range(10):
        vector_label = convert_label(i)
        if (vector == vector_label).all():
            return i
# takes largest number in vector, turns it into 1, zeros everything else,
# and converts vector to label
def label_conversion(model_labels):
    test_labels_numbers = np.zeros(len(model_labels))
    for i, k in enumerate(model_labels):
        vector = np.zeros(10)
        vector[np.where(np.abs(k) == np.amax(np.abs(k)))[0]] = 1
        model_labels[i] = vector
        test_labels_numbers[i] = convert_back(vector)
    return test_labels_numbers

# convert train labels to number vectors from equation 1 in hw6
for indx, value in enumerate(train_label):
    train_label_converted[indx, :] = convert_label(value)

print('Training image dimensions: %s x %s' % (train_image.shape[0], train_image.shape[1]))
print('Test image dimensions: %s x %s' % (test_image.shape[0], test_image.shape[1]))

print('Digits: 0 1 2 3 4 5 5 6 7 8 9')
print('labels: %s' % np.unique(train_label))
print('Class distribution: %s' % np.bincount(train_label))

# plots first nine mnist training images in one figure w title
fig, axs = plt.subplots(3,3)
fig.suptitle('First nine training MNIST images')
for i in range(9):
    plt.subplot(330 + 1 + i)
    image_reshape = train_image_sample[i].reshape(28,28)
    plt.imshow(image_reshape, cmap=plt.get_cmap('gray'))
    plt.axis('off')

plt.savefig(os.getcwd() + '/firstnine_train_images.png', bbox_inches='tight')
plt.show()

# plots first nine mnist test images in one figure w title
fig, axs = plt.subplots(3,3)
fig.suptitle('First nine test MNIST images')
for i in range(9):
    plt.subplot(330 + 1 + i)
    image_reshape = test_image[i].reshape(28,28)
    plt.imshow(image_reshape, cmap=plt.get_cmap('gray'))
    plt.axis('off')

plt.savefig(os.getcwd() + '/firstnine_test_images.png', bbox_inches='tight')
plt.show()

# PINV
# Y = f(X,B)
B = train_label_converted.T @ np.linalg.pinv(train_image).T
test_labels_model = B @ test_image.T
print('test!')
test_labels_numbers = label_conversion(test_labels_model.T)
for i, k in enumerate(test_labels_model.T):
    vector = np.zeros(10)
    vector[np.where(np.abs(k) == np.amax(np.abs(k)))[0]] = 1
    test_labels_model.T[i] = vector
    test_labels_numbers[i] = convert_back(vector)

acc_diff = test_labels_numbers - test_label
accuracy_pinv = ( sum(acc_diff==0) / len(acc_diff) ) * 100
print(f'\n\nAccuracy for pinv model: {accuracy_pinv}%\n\n')

fig, axs = plt.subplots(2,1)
axs[0].bar(list(range(100)), test_labels_numbers[:100], linewidth=2)
axs[0].set_ylim(0)
axs[0].set_title(f'Accuracy = {accuracy_pinv}%')
axs[0].set_xlim(0, 100)


# plots first one-hundred labels vs ground truth
axs[1].bar(list(range(100)), test_label[:100], linewidth=2, color='r')
axs[1].set_ylim(0)
axs[1].set_xlim(0, 100)

B_image = B.reshape(10, int(np.sqrt(B.shape[1])), int(np.sqrt(B.shape[1])))

plt.savefig(os.getcwd() + '/barcharts_pinv.png', bbox_inches='tight')
plt.show()

# plots weights for each digit
fig, axs = plt.subplots(4,3)
for k in range(10):
    plt.subplot(4, 3, 1 + k)
    plt.pcolor(B_image[k])
    if k == 9:
        plt.title('0', fontsize='x-small')
    else:
        plt.title(f'{k+1}', fontsize='x-small')
    plt.gca().invert_yaxis()
    plt.axis('off')
    # plt.colorbar(coef_map, ax=axs[1])
fig.suptitle(f'Error = {np.round(accuracy_pinv, 3)}%')

plt.savefig(os.getcwd() + '/pixels_pinv.png', bbox_inches='tight')
plt.show()

# weights in lineplot form
plt.figure(figsize=(30,20))
for k in range(10):
    plt.subplot(4, 3, 1 + k)
    plt.plot(B[k], color='g', linewidth=2)
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')

    if k == 9:
        plt.title('0', fontsize='medium')
    else:
        plt.title(f'{k+1}', fontsize='medium')

plt.savefig(os.getcwd() + '/pixels_linechart_pinv.png', bbox_inches='tight')
plt.show()

'''
# LASSO
'''
errors_for_lasso = []
# iterates through different lambdas
for i in alpha:
    lasso = Lasso(alpha=i) # intiialize lasso instance with initial lambda = 0.1

    # fit lasso model
    fit = lasso.fit(train_image, train_label_converted)
    pred = lasso.predict(test_image)

    print(f'\nlambda = {i}')

    B = lasso.coef_
    test_labels_model = B @ test_image.T
    test_labels_numbers = label_conversion(test_labels_model.T)

    acc_diff = test_labels_numbers - test_label
    accuracy_lasso = ( sum(acc_diff==0) / len(acc_diff) ) * 100
    errors_for_lasso.append(accuracy_lasso)

    print(f'Accuracy for lasso model: {np.round(accuracy_lasso, 3)}%\n\n')

# plot accuracies as function of lambda
plt.scatter(alpha, errors_for_lasso, color='k')
plt.savefig(os.getcwd() + '/accuracies_lasso_linechart.png', bbox_inches='tight')
'''
    # plots first one-hundred labels vs ground truth
    fig, axs = plt.subplots(2,1)
    axs[0].bar(list(range(100)), test_labels_numbers[:100], linewidth=2)
    axs[0].set_title(f'lambda = {i}, Error = {np.round(accuracy_lasso, 3)}%')

    axs[1].bar(list(range(100)), test_label[:100], color='r', linewidth=2)
    plt.savefig(os.getcwd() + f'/labels_lasso_lambda{i}.png', bbox_inches='tight')
    plt.show()

    # pixel weights in lineplot form
    plt.figure(figsize=(30,20))
    for k in range(10):
        plt.subplot(4, 3, 1 + k)
        plt.plot(B[k], color='g', linewidth=2)
        plt.xticks(fontsize='small')
        plt.yticks(fontsize='small')

        if k == 9:
            plt.title('0', fontsize='medium')
        else:
            plt.title(f'{k+1}', fontsize='medium')

        plt.savefig(os.getcwd() + f'/pixels_linechart_lasso_lambda{i}.png', bbox_inches='tight')
        plt.show()

        B_image = B.reshape(10, int(np.sqrt(B.shape[1])), int(np.sqrt(B.shape[1])))

    #   plots pixel weights for each digit
        fig, axs = plt.subplots(4,3)
        for k in range(10):
            plt.subplot(4, 3, 1 + k)
            plt.pcolor(B_image[k])
            if k == 9:
                plt.title('0', fontsize='x-small')
            else:
                plt.title(f'{k+1}', fontsize='x-small')
            plt.gca().invert_yaxis()
            plt.axis('off')

        fig.suptitle(f'lambda = {i}, Error = {np.round(accuracy_lasso, 3)}%')
        plt.savefig(os.getcwd() + f'/pixels_lasso_lambda{i}.png', bbox_inches='tight')
        plt.show()
'''
'''
# ElasticNet
'''

# training model
e_net = ElasticNet(alpha=alpha[1]) # lambda = 0.2

# fit e_net model
fit = e_net.fit(train_image, train_label_converted)
pred = e_net.predict(test_image)

print(f'lambda = {alpha[1]}')

B = e_net.coef_
test_labels_model = B @ test_image.T

test_labels_numbers = label_conversion(test_labels_model.T)

acc_diff = test_labels_numbers - test_label
accuracy_enet = ( sum(acc_diff==0) / len(acc_diff) ) * 100
print(f'Accuracy for ElasticNet model: {np.round(accuracy_enet, 3)}%\n\n')

# apply most important pixels to test images to see if I get the same label
accuracy = []
percents = np.arange(49, 100, 5)
for i in percents:
    optimize_b = np.copy(B)
    percentile = np.percentile(B, i)
    optimize_b[np.where(optimize_b < percentile)] = 0

    test_labels_most = optimize_b @ test_image.T
    test_labels_most_numbers = label_conversion(test_labels_most.T)

    acc_diff1 = test_labels_most_numbers - test_label
    accuracy1 = (sum(acc_diff1 == 0) / len(acc_diff1))  * 100
    accuracy.append(accuracy1)

    print(f'percentile: {i}')
    print(f'Accuracy of threshold: {accuracy1}%')

plt.scatter(percents, accuracy, color='k')
plt.ylim(0,100)
plt.xlabel('Percentile')
plt.ylabel('Accuracy (%)')
plt.title('All digits')
plt.grid()
plt.savefig(os.getcwd() + '/percentile_vs_acc_enet.png', bbox_inches='tight')
plt.show()

# now do this for each digit

# list of percentiles
percents = np.arange(49, 100, 5)
for r in range(10):
    number_present = np.where(test_label == r)
    number = test_label[number_present]
    number_images = test_image[number_present]

    accuracy = []
    for i in percents:
        optimize_b = np.copy(B)
        percentile = np.percentile(B, i)
        optimize_b[np.where(optimize_b < percentile)] = 0
        test_labels_most = optimize_b @ number_images.T

        test_labels_most_numbers = label_conversion(test_labels_most.T)
        acc_diff1 = test_labels_most_numbers - number
        accuracy1 = (sum(acc_diff1 == 0) / len(acc_diff1))  * 100
        accuracy.append(accuracy1)

        print(f'percentile: {i}')
        print(f'Accuracy of threshold: {accuracy1}%')

    plt.scatter(percents, accuracy, color='k')
    plt.ylim(0,100)
    plt.xlabel('Percentile')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{r}')
    plt.grid()
    plt.savefig(os.getcwd() + f'/percentile_vs_acc_enet{r}.png', bbox_inches='tight')
    plt.show()

errors_for_enet = []
for i in alpha:
    e_net = ElasticNet(alpha=i) # intiialize e_net instance with initial lambda

    # fit lasso model
    fit = e_net.fit(train_image, train_label_converted)
    pred = e_net.predict(test_image)

    print(f'lambda = {i}')

    B = e_net.coef_
    test_labels_model = B @ test_image.T

    test_labels_numbers = label_conversion(test_labels_model.T)

    acc_diff = test_labels_numbers - test_label
    accuracy_enet = ( sum(acc_diff==0) / len(acc_diff) ) * 100

    errors_for_enet.append(accuracy_enet)

    print(f'Accuracy for ElasticNet model: {np.round(accuracy_enet, 3)}%\n\n')

plt.scatter(alpha, errors_for_enet, color='k')
plt.savefig(os.getcwd() + '/accuracies_enet_linechart.png', bbox_inches='tight')
'''
    axs[0].bar(list(range(100)), test_labels_numbers[:100], linewidth=2)
    # axs[0].invert_yaxis()
    # axs[0].set_axis('off')
    axs[0].set_title(f'lambda = {i}, Error = {np.round(accuracy_enet, 3)}%')


    axs[1].bar(list(range(100)), test_label[:100], color='r', linewidth=2)
    # axs[1].invert_yaxis()
    # axs[1].set_axis('off')
    # axs[1].set_title(f'lambda = {i}', fontsize='small')
    plt.savefig(os.getcwd() + f'/labels_enet_lambda{i}.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(30,20))
    for k in range(10):
        plt.subplot(4, 3, 1 + k)
        plt.plot(B[k], color='g', linewidth=2)
        plt.xticks(fontsize='small')
        plt.yticks(fontsize='small')

        if k == 9:
            plt.title('0', fontsize='medium')
        else:
            plt.title(f'{k+1}', fontsize='medium')
        # plt.gca().invert_yaxis()
        # plt.axis('off')
        # plt.colorbar(coef_map, ax=axs[1])
    # fig.suptitle(f'lambda = {i}, Error = {np.round(accuracy_enet, 3)}%')

    # axs[1].set_xlim(0, 800)
    plt.savefig(os.getcwd() + f'/pixels_linechart_enet_lambda{i}.png', bbox_inches='tight')
    plt.show()

    B_image = B.reshape(10, int(np.sqrt(B.shape[1])), int(np.sqrt(B.shape[1])))

    fig, axs = plt.subplots(4,3)
    for k in range(10):
        plt.subplot(4, 3, 1 + k)
        plt.pcolor(B_image[k])
        if k == 9:
            plt.title('0', fontsize='x-small')
        else:
            plt.title(f'{k+1}', fontsize='x-small')
        plt.gca().invert_yaxis()
        plt.axis('off')
        # plt.colorbar(coef_map, ax=axs[1])
    fig.suptitle(f'lambda = {i}, Error = {np.round(accuracy_enet, 3)}%')

    plt.savefig(os.getcwd() + f'/pixels_enet_lambda{i}.png', bbox_inches='tight')
    plt.show()
'''
'''
    axs[0].bar(list(range(test_labels_model.shape[1])), test_labels_model[0], linewidth=2)
    # axs[0].invert_yaxis()
    # axs[0].set_axis('off')
    axs[0].set_title(f'lambda = {i}', fontsize='small')

    B_image = B.reshape(int(np.sqrt(B.shape[0])), int(np.sqrt(B.shape[0])))
    # print(B_image.shape)
    coef_map = axs[1].pcolor(B_image)
    axs[1].invert_yaxis()
    plt.colorbar(coef_map, ax=axs[1])

    axs[2].bar(range(len(B)), B[:,0], linewidth=2)
    # axs[1].set_xlim(0, 800)
    plt.savefig(os.getcwd() + f'/enet_lambda{i}.png', bbox_inches='tight')
    plt.show()
'''
'''Ridge'''
errors_for_ridge = []
for i in alpha:
    ridgeR = Ridge(alpha=i) # intiialize e_net instance with initial lambda
    # fig, axs = plt.subplots(2,1)

    # fit lasso model
    fit = ridgeR.fit(train_image, train_label_converted)
    pred = ridgeR.predict(test_image)

    # calculate mean square error
    # mse_ridge = np.mean((pred - test_label)**2)
    print(f'lambda = {i}')
    # print(f'MSE = {mse_ridge}\n\n')

    B = ridgeR.coef_
    test_labels_model = B @ test_image.T
    # print(test_labels_model.shape)

    test_labels_numbers = label_conversion(test_labels_model.T)

    acc_diff = test_labels_numbers - test_label
    accuracy_ridge = ( sum(acc_diff==0) / len(acc_diff) ) * 100

    errors_for_ridge.append(accuracy_ridge)

    print(f'Accuracy for Ridge model: {np.round(accuracy_ridge, 3)}%\n\n')

plt.scatter(alpha, errors_for_ridge, color='k')
plt.savefig(os.getcwd() + '/accuracies_ridge_linechart.png', bbox_inches='tight')
'''
    axs[0].bar(list(range(100)), test_labels_numbers[:100], linewidth=2)
    # axs[0].invert_yaxis()
    # axs[0].set_axis('off')
    axs[0].set_title(f'lambda = {i}, Error = {np.round(accuracy_ridge, 3)}%')


    axs[1].bar(list(range(100)), test_label[:100], color='r', linewidth=2)
    # axs[1].invert_yaxis()
    # axs[1].set_axis('off')
    # axs[1].set_title(f'lambda = {i}', fontsize='small')
    plt.savefig(os.getcwd() + f'/labels_ridge_lambda{i}.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(30,20))
    for k in range(10):
        plt.subplot(4, 3, 1 + k)
        plt.plot(B[k], color='g', linewidth=2)
        plt.xticks(fontsize='small')
        plt.yticks(fontsize='small')

        if k == 9:
            plt.title('0', fontsize='medium')
        else:
            plt.title(f'{k+1}', fontsize='medium')
        # plt.gca().invert_yaxis()
        # plt.axis('off')
        # plt.colorbar(coef_map, ax=axs[1])
    # fig.suptitle(f'lambda = {i}, Error = {np.round(accuracy_enet, 3)}%')

    # axs[1].set_xlim(0, 800)
    plt.savefig(os.getcwd() + f'/pixels_linechart_ridge_lambda{i}.png', bbox_inches='tight')
    plt.show()

    B_image = B.reshape(10, int(np.sqrt(B.shape[1])), int(np.sqrt(B.shape[1])))

    fig, axs = plt.subplots(4,3)
    for k in range(10):
        plt.subplot(4, 3, 1 + k)
        plt.pcolor(B_image[k])
        if k == 9:
            plt.title('0', fontsize='x-small')
        else:
            plt.title(f'{k+1}', fontsize='x-small')
        plt.gca().invert_yaxis()
        plt.axis('off')
        # plt.colorbar(coef_map, ax=axs[1])
    fig.suptitle(f'lambda = {i}, Error = {np.round(accuracy_ridge, 3)}%')

    plt.savefig(os.getcwd() + f'/pixels_ridge_lambda{i}.png', bbox_inches='tight')
    plt.show()
'''
