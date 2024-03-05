import cv2

def findFace(strm):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    strmGray = cv2.cvtColor(strm, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(strmGray, scaleFactor=1.2,minNeighbors=8)

    faceArray = []
    faceAreaArray = []

    for x,y,width,height in faces:
        cv2.rectangle(strm, (x,y), (x + width, y + height), (255,0,255), 2)
        centerX = x + (width // 2)
        centerY = y + (height // 2)
        area = width * height
        cv2.circle(strm, (centerX, centerY), 5, (0, 0, 0), cv2.FILLED)
        faceArray.append([centerX, centerY])
        faceAreaArray.append(area)
    if len(faceAreaArray) != 0:
        i = faceAreaArray.index(max(faceAreaArray))
        return strm, [faceArray[i], faceAreaArray]
    else:
        return strm, [[0,0],0]


cap = cv2.VideoCapture(0)

while True:
    _, strm = cap.read()
    strm, info = findFace(strm)
    print("Center: ", info[0], "Area: ", info[1])
    cv2.imshow("Stream", strm)
    cv2.waitKey(1)