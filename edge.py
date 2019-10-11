# import cv2
# # import numpy as np

# cap = cv2.VideoCapture(0)

# while True:
# 	uselessData, frame = cap.read() #first one is useless data, this is actually called a 3D list

# 	edges = cv2.Canny(frame, 50, 50) 
# 	# ColorizedEdges = np.zeros((480, 640, 3), dtype=np.unit8)

# 	# for row in range(len(edges)):
# 	# 	for col in range(len(edges[row])):
# 	# 		if edges[row][col]:
# 	# 			ColorizedEdges[row][col] = frame[row][col]

# 	# cv2.imshow('ColorizedEdges Edges', ColorizedEdges)

# 	cv2.imshow("Edges", edges) 
    
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# cv2.destroyAllWindow()
# cap.release()

import cv2
import numpy as np
import argparse
import sys

 

parser = argparse.ArgumentParser()
parser.add_argument("--camera", "-c", type=int, default=-1,
                    help="Set which camera should be used")
args = parser.parse_args()

 

if args.camera == -1:
    print("Use '--camera' or '-c' to choose the camera you want to use.")
    indexes = []
    for ii in range(20):
        cap = cv2.VideoCapture(ii)
        if cap.isOpened():
            indexes.append(ii)

 

    print("Here are the cameras we found:")
    for index in indexes:
        print(index, end=', ')

 

    sys.exit()

 


def main():
    cap = cv2.VideoCapture(args.camera)

 

    while cap.isOpened():  # loop runs if capturing has been initialized
        ret, frame = cap.read()  # reads frames from a camera
        edges = cv2.Canny(frame, 50, 50)  # finds edges in the input image
        colEdges = np.zeros((len(frame), len(frame[0]), 3), dtype=np.uint8)

 

        for row in range(len(edges)):
            for col in range(len(edges[row])):
                if edges[row][col]:
                    colEdges[row][col] = frame[row][col]

 

        cv2.imshow('Colorized Edges', colEdges)

 

        # Wait for Esc key to stop
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cap.release()  # Close the window

 


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()  # De-allocate any associated memory usage