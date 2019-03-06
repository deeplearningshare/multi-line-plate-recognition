import numpy as np
import cv2
import pickle

# create train label
allnums = 5
a = np.loadtxt('./data/label_train.txt')
b = np.zeros([allnums, 65])
for i in range(allnums):
    for j in range(7):
        b[i, int(a[i, j])] = int(a[i, j])

# create image train data
path = './data/train_data'
img_data = np.zeros([allnums, 64, 96, 3])
for i in range(allnums):
    img_path = path + '/' + str(i).zfill(5) + ".jpg"
    b = cv2.imread(img_path)
    img_data[i, :, :, :] = b
    if i%1000==0:
        print('num:' + str(i))
np.save("./data/pp_train.npy",img_data)
# mg_data = np.load("./data/pp_train.npy")
# mg_data1 = mg_data.transpose("./data/pp_train.npy")
print("finish")
