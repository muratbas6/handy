from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# dimensions of our images
img_width, img_height = 100,100






model = load_model('hand_model.h5')
model.load_weights('hand_weights.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])





import cv2

cap = cv2.VideoCapture(0)
i = 0
while 1:
    ret, img = cap.read()
    cv2.rectangle(img, (50, 50), (350,350), (255, 0, 0), 2)
    capture = img[50:350,50:350]
    resized_image = cv2.resize(capture, (100, 100))



    cv2.imwrite("bunagozat.jpg",resized_image)
    x = image.img_to_array(resized_image)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    print(classes)
    font =  "CV_FONT_HERSHEY_SIMPLEX"
    cv2.putText(img,str(classes), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
    cv2.imshow('img', img)
    cv2.waitKey(100)

    if 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
