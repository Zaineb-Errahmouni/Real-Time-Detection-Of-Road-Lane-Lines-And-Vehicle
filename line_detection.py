
import cv2
import numpy as np
import datetime

car_cascade = cv2.CascadeClassifier('cars.xml')
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
     for line in lines:
        for x1, y1, x2, y2 in line:
             cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=5)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

# = cv2.imread('road.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [(0, height),(width/2, height/2),(width, height)]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,rho=3,theta=np.pi/180,threshold=260,lines=np.array([]),minLineLength=70,maxLineGap=1)
    image_with_lines = drow_the_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # describe the type of
        # font you want to display
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
 
        # Get date and time and
        # save it inside a variable
        dt = str(datetime.datetime.now())
 
        # put the dt variable over the
        # video frame
    frame = cv2.putText(frame, dt,(700, 25),font, 1,(0, 255, 255),2, cv2.LINE_8)
    frame = process(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)
    # if str(np.array(cars).shape[0]) == '1':
    #     i += 1
    #     continue
    for (x,y,w,h) in cars:
        plate = frame[y:y + h, x:x + w]
        cv2.rectangle(frame,(x,y),(x +w, y +h) ,(51 ,51,255),2)
        cv2.putText(frame, 'Car', (x+10, y +25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.imshow('car',plate)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,217),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
    frame = process(frame)
    cv2.putText(frame,  'detection de voie :',  (50, 25),  font, 0.6,  (0, 255, 255), 1, cv2.LINE_4)
    cv2.putText(frame,  'Distance de securite :',  (50, 50),  font, 0.6,  (0, 255, 255), 1, cv2.LINE_4)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        break


cv2.destroyAllWindows()