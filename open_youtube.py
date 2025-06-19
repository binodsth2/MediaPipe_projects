import cv2
import mediapipe as mp
import webbrowser
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables for gesture detection
palm_shown = False
last_trigger_time = 0
cooldown = 2  # Cooldown in seconds to prevent multiple triggers

def detect_palm(hand_landmarks):
    # Get fingertip y-coordinates
    fingertips = [hand_landmarks.landmark[i].y for i in [8, 12, 16, 20]]  # Index, Middle, Ring, Pinky
    palm_point = hand_landmarks.landmark[0].y  # Wrist point
    
    # Check if all fingertips are above the palm point and roughly aligned
    fingers_up = all(tip < palm_point for tip in fingertips)
    max_y_diff = max(fingertips) - min(fingertips)
    
    return fingers_up and max_y_diff < 0.1

def open_youtube():
    # Replace with your desired YouTube video URL
    video_url = "https://youtu.be/CxAWKewvooo?si=Bq1NJO9b7omFPMkP"
    webbrowser.open(video_url)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally
    image = cv2.flip(image, 1)
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Check for palm gesture
            current_time = time.time()
            if detect_palm(hand_landmarks):
                if not palm_shown and (current_time - last_trigger_time) > cooldown:
                    print("Palm detected! Opening YouTube...")
                    open_youtube()
                    last_trigger_time = current_time
                palm_shown = True
            else:
                palm_shown = False

    # Add instruction text
    cv2.putText(image, "Show palm to open YouTube", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Control', image)
    
    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()