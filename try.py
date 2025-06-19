import cv2
import mediapipe as mp
import numpy as np
import random
import pyttsx3

# Window and layout settings
WINDOW_WIDTH, WINDOW_HEIGHT = 1600, 920
BLOCK_SIZE = 50
BUCKET_CAPACITY = 4

# Button coordinates
NEXT_BTN = {
    'x1': WINDOW_WIDTH - 200,
    'y1': WINDOW_HEIGHT - 100,
    'x2': WINDOW_WIDTH - 50,
    'y2': WINDOW_HEIGHT - 30
}

QUIT_BTN = {
    'x1': 50,
    'y1': WINDOW_HEIGHT - 100,
    'x2': 200,
    'y2': WINDOW_HEIGHT - 30
}

BUCKET = {
    'x1': 700,
    'y1': 250,
    'x2': 900,
    'y2': 650
}

def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    return mp_hands, hands, mp_drawing

def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

def draw_3d_block(img, x, y, size=50, color=(0, 0, 255), val=None):
    # Front face
    cv2.rectangle(img, (x, y), (x + size, y + size), color, -1)
    
    # 3D effect offset
    offset = int(size * 0.3)
    
    # Draw 3D edges
    pts = np.array([
        [x + offset, y - offset],
        [x + size + offset, y - offset],
        [x + size + offset, y + size - offset],
        [x + offset, y + size - offset]
    ], np.int32)
    
    # Draw shaded sides
    cv2.fillPoly(img, [pts], (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7)))
    
    # Draw block value
    if val is not None:
        cv2.putText(img, str(val), (x + size//3, y + int(size*0.7)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

def draw_button(img, btn, text, color=(0, 128, 255)):
    cv2.rectangle(img, (btn['x1'], btn['y1']), (btn['x2'], btn['y2']), color, -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = btn['x1'] + (btn['x2'] - btn['x1'] - text_size[0]) // 2
    text_y = btn['y1'] + (btn['y2'] - btn['y1'] + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

def generate_question():
    op = random.choice(['+', '-'])
    a = random.randint(0, 4)
    b = random.randint(0, 4) if op == '+' else random.randint(0, a)
    answer = a + b if op == '+' else a - b
    
    question = f"{a} {op} {b} = ?"
    options = [answer]
    while len(options) < 3:
        wrong = random.randint(0, 8)
        if wrong not in options:
            options.append(wrong)
    random.shuffle(options)
    return question, options, answer

def reset_blocks():
    return [{'pos': (150 + i * 70, 600), 'val': i, 'picked': False, 'moving': False} 
            for i in range(5)], []

def inside_bucket(x, y):
    return (BUCKET['x1'] <= x <= BUCKET['x2'] - BLOCK_SIZE and 
            BUCKET['y1'] <= y <= BUCKET['y2'] - BLOCK_SIZE)

def main():
    mp_hands, hands, mp_drawing = init_mediapipe()
    engine = init_tts()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    
    cv2.namedWindow("Gesture Math Box", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Math Box", WINDOW_WIDTH, WINDOW_HEIGHT)
    
    blocks, moved_blocks = reset_blocks()
    question, options, answer = generate_question()
    option_pos = [(150, 150), (350, 150), (150, 250)]
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
            
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Process hand landmarks
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Draw UI elements
        draw_button(img, NEXT_BTN, "Next")
        draw_button(img, QUIT_BTN, "Quit", color=(0, 0, 255))
        cv2.rectangle(img, (BUCKET['x1'], BUCKET['y1']), 
                     (BUCKET['x2'], BUCKET['y2']), (0, 128, 255), 4)
        
        # Draw question and options
        cv2.rectangle(img, (0, 0), (500, 65), (0, 255, 0), -1)
        cv2.putText(img, question, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        for val, pos in zip(options, option_pos):
            cv2.circle(img, pos, 40, (255, 255, 255), -1)
            cv2.putText(img, str(val), (pos[0] - 20, pos[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Handle hand tracking and interactions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                ix = int(hand_landmarks.landmark[8].x * WINDOW_WIDTH)
                iy = int(hand_landmarks.landmark[8].y * WINDOW_HEIGHT)
                
                # Draw finger position
                cv2.circle(img, (ix, iy), 15, (0, 0, 255), -1)
                
                # Handle block interactions
                for block in blocks:
                    if not block['picked'] and not block['moving']:
                        bx, by = block['pos']
                        if (bx < ix < bx + BLOCK_SIZE and 
                            by < iy < by + BLOCK_SIZE):
                            block['moving'] = True
                    
                    if block['moving']:
                        draw_3d_block(img, ix - BLOCK_SIZE//2, 
                                    iy - BLOCK_SIZE//2, val=block['val'])
                        if inside_bucket(ix, iy) and len(moved_blocks) < BUCKET_CAPACITY:
                            moved_blocks.append({'val': block['val']})
                            block['moving'] = False
                            block['picked'] = False
                
                # Handle button clicks
                if (NEXT_BTN['x1'] <= ix <= NEXT_BTN['x2'] and 
                    NEXT_BTN['y1'] <= iy <= NEXT_BTN['y2']):
                    blocks, moved_blocks = reset_blocks()
                    question, options, answer = generate_question()
                
                if (QUIT_BTN['x1'] <= ix <= QUIT_BTN['x2'] and 
                    QUIT_BTN['y1'] <= iy <= QUIT_BTN['y2']):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        
        # Draw blocks
        for block in blocks:
            if not block['picked'] and not block['moving']:
                draw_3d_block(img, block['pos'][0], block['pos'][1], 
                            val=block['val'])
        
        # Draw moved blocks in bucket
        for idx, block in enumerate(moved_blocks):
            bucket_x = BUCKET['x1'] + 20
            bucket_y = BUCKET['y2'] - BLOCK_SIZE - idx * (BLOCK_SIZE + 10)
            draw_3d_block(img, bucket_x, bucket_y, color=(0, 255, 0), 
                         val=block['val'])
        
        cv2.imshow("Gesture Math Box", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()