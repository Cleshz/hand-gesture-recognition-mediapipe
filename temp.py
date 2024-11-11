import cv2
import mediapipe as mp
from djitellopy import Tello
import time
import threading

# Initialize Mediapipe and Tello
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
tello = Tello()
tello.connect()

# Check drone battery level
print(f"Battery level: {tello.get_battery()}%")

# Track drone state and timing to avoid repeated commands
drone_state = "landed"  # Possible states: "landed", "flying"
last_command_time = 0   # Tracks the last time a command was sent
command_cooldown = 2    # Cooldown period in seconds
frame = None            # Shared frame between threads
frame_lock = threading.Lock()  # Lock to protect access to the frame

# Function to capture frames in a separate thread
def capture_frames():
    global frame
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, captured_frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        with frame_lock:
            frame = cv2.flip(captured_frame, 1)  # Flip frame for selfie view
    cap.release()

# Function to send drone commands in a separate thread
def execute_command(command, *args):
    try:
        if command == 'takeoff':
            tello.takeoff()
        elif command == 'land':
            tello.land()
        elif command == 'move_left':
            tello.move_left(*args)
        elif command == 'move_right':
            tello.move_right(*args)
    except Exception as e:
        print(f"An error occurred while executing {command}: {e}")

# Start video capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Initialize Mediapipe Hands
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        # Get a frame from the capture thread
        with frame_lock:
            if frame is None:
                continue
            image = frame.copy()

        # Convert the image to RGB for Mediapipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Get the current time to manage cooldown
        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Calculate distances for gestures
                thumb_index_dist = abs(thumb_tip.y - index_tip.y)
                index_middle_dist = abs(index_tip.y - middle_tip.y)
                index_ring_dist = abs(index_tip.y - ring_tip.y)
                index_pinky_dist = abs(index_tip.y - pinky_tip.y)

                try:
                    # Recognize "thumbs-up" gesture (for takeoff)
                    if thumb_index_dist > 0.1 and index_pinky_dist < 0.1 and drone_state == "landed" and (current_time - last_command_time) > command_cooldown:
                        cv2.putText(image, 'Takeoff', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        command_thread = threading.Thread(target=execute_command, args=('takeoff',))
                        command_thread.start()
                        drone_state = "flying"
                        last_command_time = current_time

                    # Recognize "full palm" gesture (for landing)
                    elif index_pinky_dist < 0.1 and drone_state == "flying" and (current_time - last_command_time) > command_cooldown:
                        cv2.putText(image, 'Landing', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        command_thread = threading.Thread(target=execute_command, args=('land',))
                        command_thread.start()
                        drone_state = "landed"
                        last_command_time = current_time

                    # Recognize "move left" gesture (only index finger extended)
                    elif index_middle_dist > 0.1 and index_ring_dist < 0.1 and index_pinky_dist < 0.1 and drone_state == "flying" and (current_time - last_command_time) > command_cooldown:
                        cv2.putText(image, 'Move Left', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        command_thread = threading.Thread(target=execute_command, args=('move_left', 30))
                        command_thread.start()
                        last_command_time = current_time

                    # Recognize "move right" gesture (index and middle fingers extended)
                    elif index_middle_dist < 0.1 and index_ring_dist > 0.1 and index_pinky_dist > 0.1 and drone_state == "flying" and (current_time - last_command_time) > command_cooldown:
                        cv2.putText(image, 'Move Right', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                        command_thread = threading.Thread(target=execute_command, args=('move_right', 30))
                        command_thread.start()
                        last_command_time = current_time

                except Exception as e:
                    print(f"An exception occurred: {e}")

        # Display the annotated image
        cv2.imshow('Gesture Controlled Drone', image)

        # Press 'Esc' to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

cv2.destroyAllWindows()
tello.end()
