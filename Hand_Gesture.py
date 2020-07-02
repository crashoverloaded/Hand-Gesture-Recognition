import cv2
import numpy as np

cap = cv2.VideoCapture(0)
def nothing(x):
	pass
cv2.namedWindow('Tracking')
cv2.createTrackbar("Min Value" , "Tracking" , 0 ,255, nothing)
cv2.createTrackbar("Max Value" ,"Tracking" , 0 ,255 , nothing)
while cap.isOpened():
	_ , frame = cap.read()
	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

	min_val = cv2.getTrackbarPos("Min Value" , "Tracking")
	max_val = cv2.getTrackbarPos("Max Value" , "Tracking")

	# Thresholding
	ret , thresh = cv2.threshold(gray , min_val, max_val , cv2.THRESH_BINARY)
	
	# Contours
	contours , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

	# Convex Hull 
	hull = [cv2.convexHull(contour) for contour in contours]	

	# Final Output 
	cv2.drawContours(gray , hull , -1 , (0,255,0) , 3 )
	cv2.imshow('Hand' , frame)
	cv2.imshow('Thres' , thresh)
	cv2.imshow('Hand' , gray)
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
