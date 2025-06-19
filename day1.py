import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Try to open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Setup the Pose model
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Empty frame. Skipping...")
            continue

        # Flip the image horizontally for a mirror effect
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image and get pose landmarks
        results = pose.process(image_rgb)

        # Draw the pose annotation on the image
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # Show the image in a window
        window_name = "MediaPipe Pose - Close this window to exit"
        cv2.imshow(window_name, image)

        # Exit when the window is closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
