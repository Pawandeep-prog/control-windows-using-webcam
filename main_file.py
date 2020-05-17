import cv2
import numpy as np

from tensorflow.keras.models import load_model


model = load_model('finger.h5')
res_ = ['zero', 'one', 'two', 'three', 'four', 'five']


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.blur(frame, (4,4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0,66,24])
    upper = np.array([179,255,255])
    #lower = np.array([38,86,0])
    #upper = np.array([121,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = mask[20:200, 240:420]
#    _, contors, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    
    
#    cv2.drawContours(mask, contors, -1, (255), 10)
    mask = cv2.resize(mask, (128,128))

    mask = mask.reshape(-1,128,128,1)

    
    
    ############################
 
    result=model.predict(mask)
    res = np.argmax(result)
    cv2.putText(frame, res_[res],(125,125), cv2.FONT_HERSHEY_SIMPLEX, 
                2,(255,0,0),2)
    cv2.rectangle(frame, (200,20), (420,240), (0,0,0), 2)

    ############################
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.resize(gray, (128,128))
    gray = gray.reshape(-1,128,128,1)
    _, contors, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    
    
    cv2.drawContours(frame, contors, -1, (0,0,255))
    '''   
    cv2.imshow("main", frame)
    cv2.imshow("result", mask.reshape(128,128))

    
    k= cv2.waitKey(1)
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()











