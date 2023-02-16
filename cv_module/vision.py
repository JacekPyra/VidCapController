import cv2
import mediapipe as mp
import numpy as np
import sys
from cv_module.cv_settings import *
import cv_module.utils as cv_utl
from game_pad_module.gamepad import GamePad

class Vision:
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    output_pts = np.float32([
            [0, 0],
            [0, SQUARE_SIZE-1],
            [SQUARE_SIZE-1, SQUARE_SIZE-1],
            [SQUARE_SIZE-1, 0]
        ])

    gamepad = GamePad()

    def find_edges(self):


        base_array = []

        for finger in ["left upper", "right upper", "right down", "left down"]:

            print(f"Point {finger} corner with your index finger tip")
            print(f"Press 'x' to continue\n\n")
            position = [0,0]



            while True:
                success, img = self.cap.read()
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(imgRGB)


                if results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, results.multi_hand_landmarks[0])
                    x = results.multi_hand_landmarks[0].landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].x
                    y = results.multi_hand_landmarks[0].landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].y

                    x = int(x * CAM_RESOLUTION[0])
                    y = int(y * CAM_RESOLUTION[1])

                    sys.stdout.write("\033[F") # Cursor up one line
                    print(f"x_pixel: {x}; y_pixel: {y}")


                    position[0] = x
                    position[1] = y

                cv2.imshow("Image", img)
                key = cv2.waitKey(100)

                if key == ord('x'):
                    break



            base_array.append(position)
            self.base_array = np.float32(base_array)

        print()
        return np.float32(base_array)
    def run(self):
        transformation_matrix = cv2.getPerspectiveTransform(self.base_array,self.output_pts)
        representation_matrix = np.zeros((SQUARE_SIZE, SQUARE_SIZE, 3))
        position = np.array([0, 0, 1])


        while True:
            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, hand)
                    x = hand.landmark[self.mpHands.HandLandmark.MIDDLE_FINGER_MCP].x
                    y = hand.landmark[self.mpHands.HandLandmark.MIDDLE_FINGER_MCP].y

                    sys.stdout.write("\033[F\033[F") # Cursor up one line
                    print(f"x_pixel: {int(x*CAM_RESOLUTION[0])}; y_pixel: {int(y*CAM_RESOLUTION[1])}")

                    position = np.array([int(x*CAM_RESOLUTION[0]), int(y*CAM_RESOLUTION[1]),1])
                    position = np.dot(transformation_matrix, position)

                    # x = cv_utl.linear_to_exponent(value=position[0])
                    # y = cv_utl.linear_to_exponent(value=position[1])

                    x = cv_utl.linear_to_limit(value=position[0])
                    y = cv_utl.linear_to_limit(value=position[1])

                    print("x_val: ", x, "y_val: ", y, sep='\t\t')
                    
                    self.gamepad.joystick(x,y)


            transformed_img = cv2.warpPerspective(img[:,:,0], transformation_matrix, (SQUARE_SIZE,SQUARE_SIZE))
            cv2.imshow("Transformed Image", transformed_img)
            try:
                representation_matrix[int(position[1])][int(position[0])][:] += 1
            except IndexError:
                pass
            cv2.imshow("Point Representation", representation_matrix)
            cv2.imshow("Image", img)
            
            cv2.waitKey(1)

