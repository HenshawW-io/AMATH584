import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import glob
import cv2
import os
from PIL import Image


# converting GIFs to PGMs
def convert_GIFtoPGM(imag):
    im = Image.open(imag)
    im = im.convert('RGB')
    return im.save(wd + 'original/yalefaces_converted/' + os.path.basename(image).replace('gif','pgm'))

no_imgs = 11

b01s = []
b02s = []
b03s = []
b04s = []
b05s = []
b06s = []
b07s = []
b08s = []
b09s = []
b10s = []
b11s = []
b12s = []
b13s = []
b14s = []
b15s = []
b16s = []
b17s = []
b18s = []
b19s = []
b20s = []
b21s = []
b22s = []
b23s = []
b24s = []
b25s = []
b26s = []
b27s = []
b28s = []
b29s = []
b30s = []
b31s = []
b32s = []
b33s = []
b34s = []
b35s = []
b36s = []
b37s = []
b38s = []
b39s = []


# print(cv2.__version__)
wd = '/home/billyh/Documents/AMATH 584: Linear Algebra/hw2_data/'

cropped_imgs = glob.glob(wd + 'cropped/CroppedYale/*/*')
original_imgs = glob.glob(wd + 'original/yalefaces/*')

# this converts images from gif to pgm
# for image in original_imgs:
#     convert_GIFtoPGM(image)

converted_imgs = glob.glob(wd + 'original/yalefaces_converted/*')
# print(cropped_imgs)

'''This categorizes the images into their subjects'''

b01s = [cv2.imread(a, 0) for a in cropped_imgs if 'B01' in a]
b02s = [cv2.imread(a, 0) for a in cropped_imgs if 'B02' in a]
b03s = [cv2.imread(a, 0) for a in cropped_imgs if 'B03' in a]
b04s = [cv2.imread(a, 0) for a in cropped_imgs if 'B04' in a]
b05s = [cv2.imread(a, 0) for a in cropped_imgs if 'B05' in a]
b06s = [cv2.imread(a, 0) for a in cropped_imgs if 'B06' in a]
b07s = [cv2.imread(a, 0) for a in cropped_imgs if 'B07' in a]
b08s = [cv2.imread(a, 0) for a in cropped_imgs if 'B08' in a]
b09s = [cv2.imread(a, 0) for a in cropped_imgs if 'B09' in a]
b10s = [cv2.imread(a, 0) for a in cropped_imgs if 'B10' in a]
b11s = [cv2.imread(a, 0) for a in cropped_imgs if 'B11' in a]
b12s = [cv2.imread(a, 0) for a in cropped_imgs if 'B12' in a]
b13s = [cv2.imread(a, 0) for a in cropped_imgs if 'B13' in a]
b15s = [cv2.imread(a, 0) for a in cropped_imgs if 'B15' in a]
b16s = [cv2.imread(a, 0) for a in cropped_imgs if 'B16' in a]
b17s = [cv2.imread(a, 0) for a in cropped_imgs if 'B17' in a]
b18s = [cv2.imread(a, 0) for a in cropped_imgs if 'B18' in a]
b19s = [cv2.imread(a, 0) for a in cropped_imgs if 'B19' in a]
b20s = [cv2.imread(a, 0) for a in cropped_imgs if 'B20' in a]
b21s = [cv2.imread(a, 0) for a in cropped_imgs if 'B21' in a]
b22s = [cv2.imread(a, 0) for a in cropped_imgs if 'B22' in a]
b23s = [cv2.imread(a, 0) for a in cropped_imgs if 'B23' in a]
b24s = [cv2.imread(a, 0) for a in cropped_imgs if 'B24' in a]
b25s = [cv2.imread(a, 0) for a in cropped_imgs if 'B25' in a]
b26s = [cv2.imread(a, 0) for a in cropped_imgs if 'B26' in a]
b27s = [cv2.imread(a, 0) for a in cropped_imgs if 'B27' in a]
b28s = [cv2.imread(a, 0) for a in cropped_imgs if 'B28' in a]
b29s = [cv2.imread(a, 0) for a in cropped_imgs if 'B29' in a]
b30s = [cv2.imread(a, 0) for a in cropped_imgs if 'B30' in a]
b31s = [cv2.imread(a, 0) for a in cropped_imgs if 'B31' in a]
b32s = [cv2.imread(a, 0) for a in cropped_imgs if 'B32' in a]
b33s = [cv2.imread(a, 0) for a in cropped_imgs if 'B33' in a]
b34s = [cv2.imread(a, 0) for a in cropped_imgs if 'B34' in a]
b35s = [cv2.imread(a, 0) for a in cropped_imgs if 'B35' in a]
b36s = [cv2.imread(a, 0) for a in cropped_imgs if 'B36' in a]
b37s = [cv2.imread(a, 0) for a in cropped_imgs if 'B37' in a]
b38s = [cv2.imread(a, 0) for a in cropped_imgs if 'B38' in a]
b39s = [cv2.imread(a, 0) for a in cropped_imgs if 'B39' in a]

subject01 = [cv2.imread(a, 0) for a in converted_imgs if 'subject01' in a]
subject02 = [cv2.imread(a, 0) for a in converted_imgs if 'subject02' in a]
subject03 = [cv2.imread(a, 0) for a in converted_imgs if 'subject03' in a]
subject04 = [cv2.imread(a, 0) for a in converted_imgs if 'subject04' in a]
subject05 = [cv2.imread(a, 0) for a in converted_imgs if 'subject05' in a]
subject06 = [cv2.imread(a, 0) for a in converted_imgs if 'subject06' in a]
subject07 = [cv2.imread(a, 0) for a in converted_imgs if 'subject07' in a]
subject08 = [cv2.imread(a, 0) for a in converted_imgs if 'subject08' in a]
subject09 = [cv2.imread(a, 0) for a in converted_imgs if 'subject09' in a]
subject10 = [cv2.imread(a, 0) for a in converted_imgs if 'subject10' in a]
subject11 = [cv2.imread(a, 0) for a in converted_imgs if 'subject11' in a]
subject12 = [cv2.imread(a, 0) for a in converted_imgs if 'subject12' in a]
subject13 = [cv2.imread(a, 0) for a in converted_imgs if 'subject13' in a]
subject14 = [cv2.imread(a, 0) for a in converted_imgs if 'subject14' in a]
subject15 = [cv2.imread(a, 0) for a in converted_imgs if 'subject15' in a]

converted_subs = [subject01, subject02, subject03, subject04, subject05, subject06, subject07, subject08, subject09, subject10, subject11, subject12, subject13, subject14, subject15]
cropped_subs = [b01s, b02s, b03s, b04s, b05s, b06s, b07s, b08s, b09s, b10s, b11s, b12s, b13s, b15s, b16s, b17s, b18s, b19s, b20s, b21s, b22s, b23s, b24s, b25s, b26s, b27s, b28s, b29s, b30s, b31s, b32s, b33s, b34s, b35s, b36s, b37s, b38s, b39s]

og_training_data = []
cropped_training_data = []
averaged_og = []
averaged_cropped = []

'''Getting the average face for each subject'''

for i in converted_subs:
    og_training_data.append(i[: int(np.round(len(i) * 0.7))])
    averaged_og.append(np.mean(og_training_data[-1], 0))

for i in cropped_subs:
    cropped_training_data.append(i[: int(np.round(len(i) * 0.7))])
    averaged_cropped.append(np.mean(cropped_training_data[-1], 0))

'''What do the average faces look like?'''
plt.figure()
plt.subplots_adjust(hspace=0.01, wspace=0.01)
for i, j in enumerate(averaged_cropped):
    plt.subplot(10,4,i+1)
    plt.imshow(np.array(j), cmap='gray')
    plt.axis('off')
plt.savefig('avg_cropped_faces.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.subplots_adjust(hspace=0.01, wspace=0.01)
for i, j in enumerate(averaged_og):
    plt.subplot(5,3,i+1)
    plt.imshow(np.array(j), cmap='gray')
    plt.axis('off')
plt.savefig('avg_original_faces.png', bbox_inches='tight')
plt.show()

cropped = []
og = []

'''Turns all images into flattened vectors'''

for i, m in enumerate(cropped_training_data):
    for row in m:
        cropped.append(np.array(row).flatten())

for i, m in enumerate(og_training_data):
    for row in m:
        og.append(np.array(row).flatten())
        # print(np.array(row).shape)

vectorized_imgs_cropped = np.array(cropped).T
vectorized_imgs_og = np.array(og).T

'''Perform SVD on vectorized images'''
u_og, s_og, v_og = la.svd(vectorized_imgs_og, full_matrices=False)
u_cropped, s_cropped, v_cropped = la.svd(vectorized_imgs_cropped, full_matrices=False)

'''Reshape face space back into original resolution'''
u_og = u_og.reshape((subject01[0].shape[0], subject01[0].shape[1], 120))
u_cropped = u_cropped.reshape((b01s[0].shape[0], b01s[0].shape[1], 1710))

'''Plot first five eigenfaces from cropped and uncropped'''

# # cropped
fig = plt.figure()

plt.subplot(231)
plt.imshow(u_cropped.T[0].T, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.imshow(u_cropped.T[1].T, cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.imshow(u_cropped.T[2].T, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.imshow(u_cropped.T[3].T, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(u_cropped.T[4].T, cmap='gray')
plt.axis('off')
plt.savefig(f'first_five_eigfaces_cropped.png', bbox_inches='tight')
plt.show()

# original
fig = plt.figure()

plt.subplot(231)
plt.imshow(u_og.T[0].T, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.imshow(u_og.T[1].T, cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.imshow(u_og.T[2].T, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.imshow(u_og.T[3].T, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(u_og.T[4].T, cmap='gray')
plt.axis('off')
plt.savefig(f'first_five_eigfaces_og.png', bbox_inches='tight')
plt.show()

plt.savefig(f'reconstructed_cropped_face{i}.png', bbox_inches='tight')
    plt.show()
#
#
for i,j in enumerate(u_og.T[:30:2]): # the transpose here is to ensure proper indexing
    plt.imshow(j.T, cmap='gray',  label='Reconstructed, Uncropped Images')
    plt.axis('off')
    plt.savefig(f'reconstructed_original_face{i}.png', bbox_inches='tight')
    plt.show()

plt.scatter(np.arange(0, len(s_og[:50])), s_og[:50]**2 / np.sum(s_og[:50]**2), label='Uncropped Images')
plt.title('s_original')
plt.savefig('s_original.png', bbox_inches='tight')
plt.show()

plt.scatter(np.arange(0, len(s_og[:50])), np.cumsum(s_og[:50]**2) / np.sum(s_og[:50]**2), color='g', label='Uncropped Images, Cum. Sum')
plt.axhline(0.95, linestyle='--', color='black')
plt.title('s_original_cumsum')
plt.savefig('s_original_cumsum.png', bbox_inches='tight')
plt.show()

plt.scatter(np.arange(0, len(s_cropped[:50])), s_cropped[:50]**2 / np.sum(s_cropped[:50]**2), label='Cropped Images')
plt.title('s_cropped')
plt.savefig('s_cropped.png', bbox_inches='tight')
plt.show()

plt.scatter(np.arange(0, len(s_cropped[:50])), np.cumsum(s_cropped[:50]**2) / np.sum(s_cropped[:50]**2), color='g', label='Cropped Images, Cum. Sum')
plt.axhline(0.95, linestyle='--', color='black')
plt.title('s_cropped_cumsum')
plt.savefig('s_cropped_cumsum.png', bbox_inches='tight')
plt.show()


'''
Projecting average faces onto the eigenvector space to get a histogram of the average
'''
u_og = u_og.reshape(u_og.shape[0]*u_og.shape[1], u_og.shape[2])
u_cropped = u_cropped.reshape(u_cropped.shape[0]*u_cropped.shape[1], u_cropped.shape[2])

fig = plt.figure()
for i, j in enumerate(averaged_og):
    plt.subplot(5,3,i+1)
    plt.bar(x = np.arange(u_og.T[1:21,0].shape[0]), height=np.matmul(j.flatten()[:, np.newaxis].T, u_og)[0][1:21])
    plt.xticks([])
    plt.yticks([])
plt.savefig('original_projections.png', bbox_inches='tight')
plt.show()

fig = plt.figure()
for i, j in enumerate(averaged_cropped):
    plt.subplot(10,4,i+1)
    plt.bar(x = np.arange(u_cropped.T[1:21,0].shape[0]), height=np.matmul(j.flatten()[:, np.newaxis].T, u_cropped)[0][1:21])
    plt.xticks([])
    plt.yticks([])
plt.savefig('cropped_projections.png', bbox_inches='tight')
plt.show()

'''
Projecting new faces onto the eigenvector space to get a histogram of the average
'''
u_og = u_og.reshape(u_og.shape[0]*u_og.shape[1], u_og.shape[2])
u_cropped = u_cropped.reshape(u_cropped.shape[0]*u_cropped.shape[1], u_cropped.shape[2])

# let's start with plotting subject1's last uncropped picture,
# reprojected onto the first eigenvector
print(u_og.T[1:21,0].shape, np.matmul(averaged_og[0].flatten()[:, np.newaxis].T, u_og)[0][1:21].shape)

fig = plt.figure()
plt.bar(x = np.arange(u_og.T[1:21,0].shape[0]), height=np.matmul(subject01[-1].flatten()[:, np.newaxis].T, u_og)[0][1:21])
plt.xticks([])
plt.yticks([])
plt.title('Subject 1, Last Pic Projection onto Eigenfaces')
plt.savefig('subject01lastpic_projections.png', bbox_inches='tight')
plt.show()

fig = plt.figure()
plt.bar(x = np.arange(u_cropped.T[1:21,0].shape[0]), height=np.matmul(b24s[-1].flatten()[:, np.newaxis].T, u_cropped)[0][1:21])
plt.xticks([])
plt.yticks([])
plt.title('b24, Last Pic Projection onto Eigenfaces')
plt.savefig('b24lastpic_cropped_projection.png', bbox_inches='tight')
plt.show()

'''
reshape projection to get back to original image
original projection * u_transpose
'''
projO = []
projC = []
for i, j in enumerate(averaged_og):
    projO.append(np.matmul(j.flatten()[:, np.newaxis].T, u_og)[0][:3])

for i, j in enumerate(averaged_cropped):
    projC.append(np.matmul(j.flatten()[:, np.newaxis].T, u_cropped)[0][:7])

reconstructed_cropped_faces = []
reconstructed_original_faces = []

plt.figure()
plt.subplots_adjust(hspace=0.01, wspace=0.01)
for i, j in enumerate(projC):
    plt.subplot(10,4,i+1)
    recon = np.matmul(j[np.newaxis, :], u_cropped.T[:7]).reshape(averaged_cropped[0].shape)
    plt.imshow(recon, cmap='gray')
    plt.axis('off')
plt.savefig('recon_avg_cropped_faces.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.subplots_adjust(hspace=0.01, wspace=0.01)
for i, j in enumerate(projO):
    plt.subplot(5,3,i+1)
    recon = np.matmul(j[np.newaxis, :], u_og.T[:3]).reshape(averaged_og[0].shape)
    plt.imshow(recon, cmap='gray')
    plt.axis('off')
plt.savefig('recon_avg_original_faces.png', bbox_inches='tight')
plt.show()

'''now onto a test image'''

testprojC = np.matmul(b24s[-1].flatten()[:, np.newaxis].T, u_cropped)[0][:7]
testprojO = np.matmul(subject01[-1].flatten()[:, np.newaxis].T, u_og)[0][:3]

recon_test_og = np.matmul(testprojO, u_og.T[:3]).reshape(averaged_og[0].shape)
recon_test_cropped = np.matmul(testprojC, u_cropped.T[:7]).reshape(averaged_cropped[0].shape)

plt.imshow(recon_test_cropped, cmap='gray')
plt.axis('off')
plt.savefig('recon_test24s_cropped.png', bbox_inches='tight')
plt.show()

plt.imshow(recon_test_og, cmap='gray')
plt.axis('off')
plt.savefig('recon_sub01_original.png', bbox_inches='tight')
plt.show()
