# <p align='center'>Skill Assessment</p>
## <p align="center">Pet Classification</p>
## Algorithm:
1. Import necessary packages.
2. Read the dataset and normalize the data.
3. Form the CNN model using the necessary layers and filters.
4. Train the model using the training dataset.
5. Evalute the model using the test data.
6. Test the model upon various new images of dogs and cats.
## Program:
```
/*
Program to implement Pet Classification
Developed by    : Y Chethan
Register Number : 212220230008
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten

X_train=np.loadtxt('input.csv',delimiter=',')
Y_train=np.loadtxt('labels.csv',delimiter=',')
X_test=np.loadtxt('input_test.csv',delimiter=',')
Y_test=np.loadtxt('labels_test.csv',delimiter=',')
X_train=X_train.reshape(len(X_train),100,100,3)
X_test=X_test.reshape(len(X_test),100,100,3)
Y_train=Y_train.reshape(len(Y_train),1)
Y_test=Y_test.reshape(len(Y_test),1)
X_test=X_test/255.0
X_train=X_train/255.0
print("shape of x_train:",X_train.shape)
print("Shape of Y_train:",Y_train.shape)
print("Shape of x_test:",X_test.shape)
print("Shape of Y_test:",Y_test.shape)

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
x=1
while(x<=2):
    model.fit(X_train,Y_train,epochs=5,batch_size=32)
    x+=1
model.evaluate(X_test,Y_test)

import cv2
images_list=["Dog "+str(num)+".jpg" for num in range(1,4)]
for image in images_list:
    img=cv2.imread(image,-1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=img/255.0
    img=cv2.resize(img,(100,100))
    img=img.reshape(1,100,100,3)
    prediction=model.predict(img)
    prediction=prediction>0.5
    if (prediction==0):
        plt.title("It is a dog")
    else:
        plt.title("It is a cat")
    image=cv2.imread(image,1)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(image)
    plt.show()
    
images_list=["Cat "+str(num)+".jpg" for num in range(6,9)]
for cat_image in images_list:
    img=cv2.imread(cat_image,-1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=img/255.0
    img=cv2.resize(img,(100,100))
    img=img.reshape(1,100,100,3)
    prediction=model.predict(img)
    prediction=prediction>0.5
    if (prediction==0):
        plt.title("It is a dog")
    else:
        plt.title("It is a cat")
    image=cv2.imread(cat_image,1)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(image)
    plt.show()
```

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

## Output:
![image](https://user-images.githubusercontent.com/75234991/172392163-1a0afe2f-9551-4437-8b93-fde3d5f36eeb.png)

![image](https://user-images.githubusercontent.com/75234991/172392176-30827666-d3d9-4b64-8c72-05be3863e083.png)
