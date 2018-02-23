import tensorflow as tf
import numpy as np
import cv2


import os
from os.path import join as pjoin

#import detect_face
#import nn4 as network





from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib
import face

sess = tf.Session()
#images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 
#                                                       image_size, 
#                                                       image_size, 3), name='input')

data_dir='./modified'#your own train folder

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
        
     
        
        data[guy] = curr_pics
        
    return data

data=load_data(data_dir)
keys=[]
for key in data.keys():
    keys.append(key)
    print('foler:{},image numbers：{}'.format(key,len(data[key])))
    
train_x=[]
train_y=[]
face_encoder = face.Encoder()
face_recognition = face.Recognition()
face_init = face.Face()

#lena = mpimg.imread('lena.jpg')
#faces=face_encoder.generate_embedding(lena)
#emb_data=np.array(faces)
#print(emb_data.shape)
##
for x in data[keys[0]]:
    
         emb_data=face_recognition.identify(x)
#       
#        
#        
         train_x.append(emb_data)
         train_y.append(0)
         print(len(train_x))
     
#
#
for y in data[keys[1]]:
         emb_data=face_recognition.identify(y)
#     emb_data=face_encoder.generate_embedding(x)
#     emb_data=np.array(emb_data)
#     print(emb_data.shape)
#     emb_data=emb_data.reshape(-1,128)
#     print("embadding",emb_data.shape)
#   
#        
        
         train_x.append(emb_data)
         train_y.append(1)
        #    
        #
print(len(train_x))
print('搞完了，样本数为：{}'.format(len(train_x)))
#train/test split
train_x=np.array(train_x)
print(train_x.shape)

#train_x=train_x.reshape(-1,128)
train_y=np.array(train_y)
print(train_x.shape)
print(train_y.shape)
        
        
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.3, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  

classifiers = knn_classifier 

model = classifiers(X_train,y_train)  
predict = model.predict(X_test)  

accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
joblib.dump(model, './model_check_point/knn_classifier.model')
  