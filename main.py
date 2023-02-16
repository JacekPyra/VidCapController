import cv2
import mediapipe as mp
import vgamepad as vg
import numpy as np
import time 
import sys



cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
gamepad = vg.VX360Gamepad()


# base_array = np.float32([
#         [150, 58], #x_pixel: 106; y_pixel: 12
#         [502, 30],
#         [565, 403],
#         [140, 464],
#     ])


SQUARE_SIZE = 140
output_pts = np.float32([
        [0, 0],
        [0, SQUARE_SIZE-1],
        [SQUARE_SIZE-1, SQUARE_SIZE-1],
        [SQUARE_SIZE-1, 0]
    ])

time.sleep(2)

def find_edges():
    base_array = []

    for finger in ["left upper", "right upper", "right down", "left down"]:

        print(f"Point {finger} corner with your index finger tip")
        print(f"Press 'x' to continue\n\n")
        position = [0,0]



        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)


            if results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, results.multi_hand_landmarks[0])
                x = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
                y = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y

                x = int(x * 640)
                y = int(y * 480)

                sys.stdout.write("\033[F") # Cursor up one line
                print(f"x_pixel: {x}; y_pixel: {y}")


                position[0] = x
                position[1] = y

            cv2.imshow("Image", img)
            key = cv2.waitKey(100)

            if key == ord('x'):
                break



        base_array.append(position)

    print()
    return np.float32(base_array)


base_array = find_edges()




def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def linear_to_exponent(value):
    EXPONENT = 1.25
    LINEAR_MAX_CONVERSION = 4096
    if 0 < value < SQUARE_SIZE:
        if value > SQUARE_SIZE/2:
            value = translate(value,  SQUARE_SIZE/2,  SQUARE_SIZE, 1, LINEAR_MAX_CONVERSION)
            return int(np.float_power(value, EXPONENT))
        else:
            value = translate(value, 0,  SQUARE_SIZE/2, LINEAR_MAX_CONVERSION, 1)

            return int(np.float_power(value, EXPONENT)) * -1
    elif 0 < value:
        return 32700
    else:
        return -32700

def transform_mouse_pad_to_numpy_surface():
    pass



transformation_matrix = cv2.getPerspectiveTransform(base_array,output_pts)



while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand)
            x = hand.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x
            y = hand.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y

            sys.stdout.write("\033[F\033[F") # Cursor up one line

            print(f"x_pixel: {int(x*640)}; y_pixel: {int(y*480)}")

            position = np.array([int(x*640), int(y*480),1])
            new_position = np.dot(transformation_matrix, position)

            x = linear_to_exponent(value=new_position[0])
            y = linear_to_exponent(value=new_position[1])

            print("x_val: ", x, "y_val: ", y, sep='\t\t')
            
            gamepad.left_joystick(x_value=x*-1, y_value=y*-1)
            gamepad.update()


    transformed_img = cv2.warpPerspective(img[:,:,0], transformation_matrix, (SQUARE_SIZE,SQUARE_SIZE))
    cv2.imshow("Transformed Image", transformed_img)
    cv2.imshow("Image", img)
    
    cv2.waitKey(1)
