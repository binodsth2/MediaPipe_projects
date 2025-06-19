import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cube size parameters
min_cube_size = 40
max_cube_size = 200

# Cube edges for drawing
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

cube_color = (255, 0, 0)  # Blue in BGR

# Camera parameters (approximate)
focal_length = 800
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # No lens distortion

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        cube_center = None
        cube_size = min_cube_size

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # Index fingertip (8)
                x1 = int(hand_landmarks.landmark[8].x * w)
                y1 = int(hand_landmarks.landmark[8].y * h)
                # Thumb tip (4)
                x2 = int(hand_landmarks.landmark[4].x * w)
                y2 = int(hand_landmarks.landmark[4].y * h)
                cube_center = (x1, y1)

                # Calculate distance between thumb tip and index fingertip
                distance = np.hypot(x2 - x1, y2 - y1)
                # Map distance to cube size
                cube_size = int(np.interp(distance, [30, 200], [min_cube_size, max_cube_size]))

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if cube_center:
            # Define cube 3D points centered at (0,0,0)
            half = cube_size / 2
            cube_points_3d = np.float32([
                [-half, -half, -half], [-half, half, -half], [half, half, -half], [half, -half, -half],
                [-half, -half, half], [-half, half, half], [half, half, half], [half, -half, half]
            ])

            # Set cube center at the fingertip
            rvec = np.zeros((3, 1))  # No rotation
            tvec = np.array([[cube_center[0]], [cube_center[1]], [800]], dtype=np.float32)  # Z=800 for perspective

            # Project 3D points to 2D image plane
            imgpts, _ = cv2.projectPoints(cube_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Draw cube edges
            for edge in cube_edges:
                pt1, pt2 = imgpts[edge[0]], imgpts[edge[1]]
                cv2.line(frame, tuple(pt1), tuple(pt2), cube_color, 3)

        cv2.imshow('3D Blue Cube Centered on Finger', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()