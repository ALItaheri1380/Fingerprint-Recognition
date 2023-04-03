import cv2
import numpy as np

fingerprint_img = cv2.imread('Original_Image.png')
gray_fingerprint_img = cv2.cvtColor(fingerprint_img, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    result = cv2.matchTemplate(gray_frame, gray_fingerprint_img, cv2.TM_CCORR_NORMED)
    similarity_score = np.max(result)
    
    cv2.putText(frame, f'Similarity Score: {similarity_score:f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()