import matplotlib.pyplot as plt
import seaborn as sns
import keras
from tensorflow.keras.models  import Sequential
#from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np
import math

labels = ['0', '1']
img_size = 224

def logGaborValue(x,y,center_scale,center_angle,scale_bandwidth,
              angle_bandwidth, logfun):
    # transform to polar coordinates
    raw, theta = getPolar(x,y)
    # if we are at the center, return 0 as in the log space
    # zero is not defined
    if raw == 0:
        return 0

    # go to log polar coordinates
    raw = logfun(raw)

    # calculate (theta-center_theta), we calculate cos(theta-center_theta) 
    # and sin(theta-center_theta) then use atan to get the required value,
    # this way we can eliminate the angular distance wrap around problem
    costheta, sintheta = math.cos(theta), math.sin(theta)
    ds = sintheta * math.cos(center_angle) - costheta * math.sin(center_angle)    
    dc = costheta * math.cos(center_angle) + sintheta * math.sin(center_angle)  
    dtheta = math.atan2(ds,dc)

    # final value, multiply the radial component by the angular one
    return math.exp(-0.5 * ((raw-center_scale) / scale_bandwidth)**2) * \
            math.exp(-0.5 * (dtheta/angle_bandwidth)**2)




def getFilter(f_0, theta_0):
    # filter configuration
    scale_bandwidth =  0.996 * math.sqrt(2/3)
    angle_bandwidth =  0.996 * (1/math.sqrt(2)) * (np.pi/number_orientations)

    # x,y grid
    extent = np.arange(-N/2, N/2 + N%2)
    x, y = np.meshgrid(extent,extent)

    mid = int(N/2)
    ## orientation component ##
    theta = np.arctan2(y,x)
    center_angle = ((np.pi/number_orientations) * theta_0) if (f_0 % 2) \
                else ((np.pi/number_orientations) * (theta_0+0.5))

    # calculate (theta-center_theta), we calculate cos(theta-center_theta) 
    # and sin(theta-center_theta) then use atan to get the required value,
    # this way we can eliminate the angular distance wrap around problem
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    ds = sintheta * math.cos(center_angle) - costheta * math.sin(center_angle)    
    dc = costheta * math.cos(center_angle) + sintheta * math.sin(center_angle)  
    dtheta = np.arctan2(ds,dc)

    orientation_component =  np.exp(-0.5 * (dtheta/angle_bandwidth)**2)

    ## frequency componenet ##
    # go to polar space
    raw = np.sqrt(x**2+y**2)
    # set origin to 1 as in the log space zero is not defined
    raw[mid,mid] = 1
    # go to log space
    raw = np.log2(raw)

    center_scale = math.log2(N) - f_0
    draw = raw-center_scale
    frequency_component = np.exp(-0.5 * (draw/ scale_bandwidth)**2)

    # reset origin to zero (not needed as it is already 0?)
    frequency_component[mid,mid] = 0

    return frequency_component * orientation_component


number_scales = 5         # scale resolution
number_orientations = 9   # orientation resolution
N = 224
constantDim = 224          # image dimensions

def getLogGaborKernal(scale, angle, logfun=math.log2, norm = True):
    # setup up filter configuration
    center_scale = logfun(N) - scale          
    center_angle = ((np.pi/number_orientations) * angle) if (scale % 2) \
                else ((np.pi/number_orientations) * (angle+0.5))
    scale_bandwidth =  0.996 * math.sqrt(2/3)
    angle_bandwidth =  0.996 * (1/math.sqrt(2)) * (np.pi/number_orientations)

    # 2d array that will hold the filter
    kernel = np.zeros((N, N))
    # get the center of the 2d array so we can shift origin
    middle = math.ceil((N/2)+0.1)-1

    # calculate the filter
    for x in range(0,constantDim):
        for y in range(0,constantDim):
            # get the transformed x and y where origin is at center
            # and positive x-axis goes right while positive y-axis goes up
            x_t, y_t = (x-middle),-(y-middle)
            # calculate the filter value at given index
            #kernel[y,x] = getFilter(1,1)
            kernel[y,x] = getFilter(1,1)
        

    # normalize the filter energy
    if norm:
        Kernel = kernel / np.sum(kernel**2)
    return kernel

def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            
                #img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                #img_arr = cv2.cvtColor(img_arr , cv2.COLOR_BGR2GRAY)
            img_arr = cv2.imread(os.path.join(path, img))
            resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
            for scale in range(5):
                train1=cv2.filter2D(resized_arr, cv2.CV_8UC3, getFilter(1,1))                      
            data.append([train1, class_num])
                
                
                
                
    #data.append([train1, class_num])        
    return np.array(data)

train = get_data('')



val = get_data('')

# Data preprocessing
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)
x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

# Data Augumentation

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False
                            )  # randomly flip images


datagen.fit(x_train)

# Define Model
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()
# Evaluating result

opt = Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
history = model.fit(x_train,y_train,epochs = 500 , validation_data = (x_val, y_val))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict(x_val)
classes_x=np.argmax(predictions,axis=1)
#predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, classes_x, target_names = ['0 (Class 0)','1 (Class 1)']))





















    
                     






























