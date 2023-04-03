import cv2
import numpy as np
from PIL import Image


with Image.open("Orgtest_Image.png") as im:

    im_resized = im.resize((im.width//2,im.height//2))
    im_resized.save('test_Image.png')


test = cv2.imread('test_Image.png')

fingerprint_img = cv2.imread("test_Image.png", cv2.IMREAD_GRAYSCALE)
fingerprint_img = cv2.equalizeHist(fingerprint_img)
_, fingerprint_img = cv2.threshold(fingerprint_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
fingerprint_img = cv2.GaussianBlur(fingerprint_img, (5, 5), 0)

cap = cv2.VideoCapture(0)


while True:

    _ , frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.equalizeHist(gray_frame)
    _, gray_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    

    result = cv2.matchTemplate(gray_frame, fingerprint_img, cv2.TM_CCORR_NORMED)
    similarity_score = np.max(result)
    
    cv2.putText(gray_frame, f'Similarity Score: {similarity_score:f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
    
    cv2.imshow('Webcam', gray_frame)
    cv2.imshow('Original Image' , test)
    
    if similarity_score > 0.9:
        print('Fingerprint matches the original image.')
        break
    
    if cv2.waitKey(1) == ord('q'):
           break

cap.release()
cv2.destroyAllWindows()