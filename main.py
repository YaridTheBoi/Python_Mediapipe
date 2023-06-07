import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def detect(frame):
    with mp_holistic.Holistic(
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=True,
        smooth_segmentation=True) as holistic:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = holistic.process(frame)
        frame.flags.writeable = True
        return results

def holistic():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _ , image = cap.read()


        results= detect(image)

        #collecting data
        try:
            left_hand = results.left_hand_landmarks.landmark
            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())

            # print(left_hand_row)
            # print('\n')
        except:
            pass


        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1), 
            mp_drawing.DrawingSpec(color=(34,255,0), thickness=1, circle_radius=1))
        

        
        mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=0), 
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=0)) 


        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2))



        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=2))
        

        
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))


        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    holistic()