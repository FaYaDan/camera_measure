import cv2
import mediapipe as mp
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_measurements(landmarks):
    arm_length = abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST]['x'] -
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]['x'])

    shoulder_length = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]['x'] -
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]['x'])

    body_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP]['x'] -
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP]['x'])

    body_height = abs(landmarks[mp_pose.PoseLandmark.NOSE]['y'] -
                      landmarks[mp_pose.PoseLandmark.LEFT_HIP]['y'])

    body_weight = abs(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]['y'] -
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]['y'])

    return arm_length, shoulder_length, body_width, body_height, body_weight


def show_measurement_results(image, arm_length, shoulder_length, body_width, body_height, body_weight):
    cv2.putText(image, f"Arm Length: {arm_length:.2f} cm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(image, f"Shoulder Length: {shoulder_length:.2f} cm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(image, f"Body Width: {body_width:.2f} cm", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(image, f"Body Height: {body_height:.2f} cm", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Measurement Results', image)


def main():
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(image)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

                height, width, _ = image.shape
                normalized_landmarks = {}
                for landmark in mp_pose.PoseLandmark:
                    normalized_landmarks[landmark] = {
                        'x': int(results.pose_landmarks.landmark[landmark].x * width),
                        'y': int(results.pose_landmarks.landmark[landmark].y * height)
                    }
                arm_length, shoulder_length, body_width, body_height, body_weight = calculate_measurements(
                    normalized_landmarks)
                
                show_measurement_results(image, arm_length, shoulder_length, body_width, body_height, body_weight)
            
            cv2.imshow('Pose Estimation', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

 