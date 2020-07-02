import cv2
import numpy as np
from numpy import math
cap = cv2.VideoCapture(0)
def nothing(x):
	pass


cv2.namedWindow("Tracking")
cv2.createTrackbar("LH","Tracking",10,255,nothing)
cv2.createTrackbar("LS","Tracking",12,255,nothing)
cv2.createTrackbar("LV","Tracking",10,255,nothing)
cv2.createTrackbar("UH","Tracking",255,255,nothing)
cv2.createTrackbar("US","Tracking",255,255,nothing)
cv2.createTrackbar("UV","Tracking",255,255,nothing)

while cap.isOpened():
	_ , frame = cap.read()
	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

	# Rectangle 
	cv2.rectangle(frame, (100,100) , (300,300) ,(0,255,0) ,0)
	crop_img = frame[100:300 , 100:300]

	# Applying Gaussian Blur
	blur = cv2.GaussianBlur(crop_img , (3,3) , 0)

	# Change Color-Space from BGR -> HSV
	hsv = cv2.cvtColor(blur , cv2.COLOR_BGR2HSV)
	
	# TRACKING
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	l_h = cv2.getTrackbarPos("LH","Tracking")
	l_s = cv2.getTrackbarPos("LS","Tracking")
	l_v = cv2.getTrackbarPos("LV","Tracking")
	u_h = cv2.getTrackbarPos("UH","Tracking")
	u_s = cv2.getTrackbarPos("US","Tracking")
	u_v = cv2.getTrackbarPos("UV","Tracking")
        # SKIN COLOR
	l_b = np.array([l_h,l_s,l_v])
	u_b = np.array([u_h,u_s,u_v])
	mask2 = cv2.inRange(hsv,l_b,u_b)
	# Create a Binary Image with where white will be skin color and rest is black
	
	# Kernel for Morphological Transformation
	kernel  = np.ones((5,5))
	
	# Dilation and Erosion
	dilation = cv2.dilate(mask2 , kernel , iterations=1)
	erosion = cv2.erode(dilation , kernel , iterations=1)

	# Applying Gaussian Blur and Threshold
	filtered = cv2.GaussianBlur(erosion , (3,3) , 0)
	_ , thresh = cv2.threshold(filtered , 80 , 255 ,0 )

	cv2.imshow('Threshold' , thresh)
	
	# Contours
	contours , _ = cv2.findContours(thresh.copy() , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

	#
	#try:
		#Find Contour with Max AREA
	contour = max(contours, key = lambda x: cv2.contourArea(x))
		
		# BOUNDING RECTANGLE
	x,y,w,h  = cv2.boundingRect(contour)
	cv2.rectangle(crop_img , (x,y) , (x+w , y+h) , (0,255,0) ,0)
		
		# CONVEX HULL
	hull = cv2.convexHull(contour)

		# Draw
	drawing = np.zeros(crop_img.shape , np.uint8) 
	cv2.drawContours(drawing , [contour] , -1 , (0,255,0) ,0) 
	cv2.drawContours(drawing , [hull] , -1 , (0,255,0) ,0) 

		# CONVEXITY DEFECTS
	hull = cv2.convexHull(contour , returnPoints=False)
	defects = cv2.convexityDefects(contour , hull)

		# Counting Defects
	count_defects = 0

	for  i in range(defects.shape[0]):
			# s- start point , e-end point , f-farthest point , d- distance
		s , e, f, d = defects[i,0]
		start = tuple(contour[s][0])
		end = tuple(contour[e][0])
		far = tuple(contour[f][0])
		a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
		b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
		c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
		angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
			
		if angle <= 90:
			count_defects+=1
			cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
		cv2.line(crop_img, start, end, [0, 255, 0], 2)
	if count_defects == 0:
		cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
	elif count_defects == 1:
		cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
	elif count_defects == 2:
		cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
	elif count_defects == 3:
		cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
	elif count_defects == 4:
		cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
	else:
		pass
	#except:
	#	pass

    # Show required images
	cv2.imshow("Gesture", frame)
	all_image = np.hstack((drawing, crop_img))
	cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
'''
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

'''
