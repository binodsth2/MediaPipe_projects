import cv2
import mediapipe as mp
import numpy as np
import random
import pyttsx3

# Set desired window size
WINDOW_WIDTH, WINDOW_HEIGHT = 1400, 800

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking rate

# Bucket coordinates and size
bucket_x1, bucket_y1 = 700, 250
bucket_x2, bucket_y2 = 900, 650
block_size = 50
bucket_capacity = 4  # Number of blocks the bucket can hold

# Button coordinates
quit_btn_x1, quit_btn_y1, quit_btn_x2, quit_btn_y2 = 50, WINDOW_HEIGHT - 100, 200, WINDOW_HEIGHT - 30
cancel_btn_x1, cancel_btn_y1, cancel_btn_x2, cancel_btn_y2 = 650, WINDOW_HEIGHT - 100, 800, WINDOW_HEIGHT - 30
next_btn_x1, next_btn_y1, next_btn_x2, next_btn_y2 = WINDOW_WIDTH - 200, WINDOW_HEIGHT - 100, WINDOW_WIDTH - 50, WINDOW_HEIGHT - 30

def draw_bucket(img):
    cv2.rectangle(img, (bucket_x1, bucket_y1), (bucket_x2, bucket_y2), (0, 128, 255), 4)
    cv2.putText(img, "BUCKET", (bucket_x1 + 20, bucket_y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 3)

def inside_bucket(x, y):
    return bucket_x1 <= x <= bucket_x2 - block_size and bucket_y1 <= y <= bucket_y2 - block_size

def speak_question_and_options(question, options):
    engine.say(question)
    for i, opt in enumerate(options):
        engine.say(f"Option {i+1}: {opt}")
    engine.runAndWait()

def generate_question():
    op = random.choice(['+', '-'])
    a = random.randint(2, 5)
    b = random.randint(1, a if op == '-' else 5)
    if op == '+':
        answer = a + b
    elif op == '-':
        answer = a - b
    elif op == '*':
        answer = a * b
    else:
        answer = a // b

    question = f"{a} {op} {b} = ?"
    options = [answer]
    while len(options) < 3:
        wrong = answer + random.choice([-2, -1, 1, 2, 3])
        if wrong != answer and wrong > 0 and wrong not in options:
            options.append(wrong)
    random.shuffle(options)
    speak_question_and_options(question, options)
    return question, options, answer

def reset_blocks():
    # Each block now has a value (1, 2, 3, 4)
    return [{'pos': (150 + i * 70, 600), 'val': i + 1, 'picked': False, 'moving': False} for i in range(4)], []

def draw_3d_block(img, x, y, size=50, color=(0, 0, 255), val=None):
    # Front face
    front_top_left = (x, y)
    front_top_right = (x + size, y)
    front_bottom_left = (x, y + size)
    front_bottom_right = (x + size, y + size)
    # Offset for 3D effect
    offset = int(size * 0.3)
    # Back face
    back_top_left = (x + offset, y - offset)
    back_top_right = (x + size + offset, y - offset)
    back_bottom_left = (x + offset, y + size - offset)
    back_bottom_right = (x + size + offset, y + size - offset)
    # Draw faces
    cv2.rectangle(img, front_top_left, front_bottom_right, color, -1)  # Front face
    pts = np.array([back_top_left, back_top_right, back_bottom_right, back_bottom_left], np.int32)
    cv2.fillPoly(img, [pts], (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7)))  # Top face
    # Draw edges
    cv2.line(img, front_top_left, back_top_left, (0,0,0), 2)
    cv2.line(img, front_top_right, back_top_right, (0,0,0), 2)
    cv2.line(img, front_bottom_left, back_bottom_left, (0,0,0), 2)
    cv2.line(img, front_bottom_right, back_bottom_right, (0,0,0), 2)
    cv2.line(img, back_top_left, back_top_right, (0,0,0), 2)
    cv2.line(img, back_top_right, back_bottom_right, (0,0,0), 2)
    cv2.line(img, back_bottom_right, back_bottom_left, (0,0,0), 2)
    cv2.line(img, back_bottom_left, back_top_left, (0,0,0), 2)
    # Draw value
    if val is not None:
        cv2.putText(img, str(val), (x + size//3, y + int(size*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

blocks, moved_blocks = reset_blocks()
question, options, answer = generate_question()
option_pos = [(150, 150), (350, 150), (150, 250)]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
solved = False
next_gesture_active = False

cv2.namedWindow("Gesture Math Box", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Math Box", WINDOW_WIDTH, WINDOW_HEIGHT)

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw question and options
    cv2.rectangle(img, (0, 0), (500, 65), (0, 255, 0), -1)
    cv2.putText(img, question, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    for val, pos in zip(options, option_pos):
        cv2.circle(img, pos, 40, (255, 255, 255), -1)
        cv2.putText(img, str(val), (pos[0] - 20, pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    draw_bucket(img)

    # Draw 3D blocks (red for remaining)
    for block in blocks:
        # Always draw blocks (never empty)
        draw_3d_block(img, block['pos'][0], block['pos'][1], size=block_size, color=(0, 0, 255), val=block['val'])

    # Draw moved blocks inside the bucket, stacked
    for idx, block in enumerate(moved_blocks):
        bucket_block_x = bucket_x1 + 20
        bucket_block_y = bucket_y2 - block_size - idx * (block_size + 10)
        draw_3d_block(img, bucket_block_x, bucket_block_y, size=block_size, color=(0, 255, 0), val=block['val'])

    # Draw Quit, Cancel, Next buttons
    cv2.rectangle(img, (quit_btn_x1, quit_btn_y1), (quit_btn_x2, quit_btn_y2), (0, 0, 255), -1)
    cv2.putText(img, "Quit", (quit_btn_x1 + 30, quit_btn_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
    cv2.rectangle(img, (cancel_btn_x1, cancel_btn_y1), (cancel_btn_x2, cancel_btn_y2), (0, 128, 255), -1)
    cv2.putText(img, "Cancel", (cancel_btn_x1 + 20, cancel_btn_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
    cv2.rectangle(img, (next_btn_x1, next_btn_y1), (next_btn_x2, next_btn_y2), (0, 128, 255), -1)
    cv2.putText(img, "Next", (next_btn_x1 + 30, next_btn_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

    next_gesture_active = False
    quit_gesture_active = False
    cancel_gesture_active = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            ix = int(handLms.landmark[8].x * w)
            iy = int(handLms.landmark[8].y * h)
            cv2.circle(img, (ix, iy), 15, (0, 0, 255), -1)

            # Pick up a block
            for block in blocks:
                bx, by = block['pos']
                if not block.get('moving', False) and bx < ix < bx + block_size and by < iy < by + block_size:
                    block['moving'] = True

            # Move block with finger
            for block in blocks:
                if block.get('moving', False):
                    draw_3d_block(img, ix - block_size//2, iy - block_size//2, size=block_size, color=(0, 0, 255), val=block['val'])
                    # Drop block in bucket
                    if inside_bucket(ix, iy) and len(moved_blocks) < bucket_capacity:
                        moved_blocks.append({'val': block['val']})
                        block['moving'] = False  # Block is always available again
                    # Drop block outside bucket
                    elif cv2.waitKey(1) & 0xFF == ord(' '):
                        block['moving'] = False

            # Check Next button gesture
            if next_btn_x1 <= ix <= next_btn_x2 and next_btn_y1 <= iy <= next_btn_y2:
                next_gesture_active = True
                cv2.rectangle(img, (next_btn_x1, next_btn_y1), (next_btn_x2, next_btn_y2), (0, 255, 0), 5)

            # Check Quit button gesture
            if quit_btn_x1 <= ix <= quit_btn_x2 and quit_btn_y1 <= iy <= quit_btn_y2:
                quit_gesture_active = True
                cv2.rectangle(img, (quit_btn_x1, quit_btn_y1), (quit_btn_x2, quit_btn_y2), (0, 255, 0), 5)

            # Check Cancel button gesture
            if cancel_btn_x1 <= ix <= cancel_btn_x2 and cancel_btn_y1 <= iy <= cancel_btn_y2:
                cancel_gesture_active = True
                cv2.rectangle(img, (cancel_btn_x1, cancel_btn_y1), (cancel_btn_x2, cancel_btn_y2), (0, 255, 0), 5)

    # Calculate sum of values in the bucket
    bucket_sum = sum(block['val'] for block in moved_blocks)

    # Check if correct answer
    if bucket_sum == answer and len(moved_blocks) > 0:
        solved = True
        idx = options.index(answer)
        cv2.circle(img, option_pos[idx], 45, (0, 255, 0), 7)
        cv2.putText(img, "Correct! Use hand to click Next.", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 3)
    else:
        solved = False

    cv2.imshow("Gesture Math Box", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    if next_gesture_active:
        blocks, moved_blocks = reset_blocks()
        question, options, answer = generate_question()
        solved = False
        cv2.waitKey(500)

    if quit_gesture_active:
        break

    if cancel_gesture_active and moved_blocks:
        moved_blocks.pop()
        cv2.waitKey(500)

cap.release()
cv2.destroyAllWindows()