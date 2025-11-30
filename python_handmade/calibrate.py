import cv2
import mediapipe as mp
import numpy as np
import socket
import struct

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils


ShowDebugScreen = False # show screen for debug value, default is close = False, open = True

 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)# UDP Send Calibrate Value to Untiy
serverAddressPort = ("127.0.0.1", 5052)

sockImg = socket.socket(socket.AF_INET, socket.SOCK_STREAM)# TCP Recive And Send Result Image To Unity 
#  Wait For Connection From Unity
sockImg.bind(("127.0.0.1", 5055))
sockImg.listen(1)
print("Waiting for connection...")
client_socket, client_address = sockImg.accept()
print(f"Connected to {client_address}")

# Webcam
# cap = cv2.VideoCapture(1)

def calibrate_threshold(hand_crop):
    hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
    v_mean = cv2.mean(hsv[:, :, 2])[0]

   
    # Use full range conditions
    if v_mean > 200:
        preset = "very_bright"
        threshold = 100
    elif v_mean > 160:
        preset = "bright"
        threshold = 90
    elif v_mean > 130:
        preset = "normal"
        threshold = 80
    elif v_mean > 100:
        preset = "dim"
        threshold = 70
    elif v_mean > 70:
        preset = "very_dim"
        threshold = 60
    else:
        preset = "dark"
        threshold = 50

    return preset, threshold

font = cv2.FONT_HERSHEY_SIMPLEX
preset, threshold = "unknown", 0

while True:
    
    length_data = client_socket.recv(4)
    if not length_data:
        break
    length = struct.unpack('I', length_data)[0]
    
    # Receive the image data from Unity
    image_data = b""
    while len(image_data) < length:
        image_data += client_socket.recv(length - len(image_data))
    
    # Convert the image data to a NumPy array and decode it
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
    frame = img

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_vals) * w)
            x_max = int(max(x_vals) * w)
            y_min = int(min(y_vals) * h)
            y_max = int(max(y_vals) * h)

            # Draw box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            try:
                hand_crop = frame[y_min:y_max, x_min:x_max]
                preset, threshold = calibrate_threshold(hand_crop)
                
                sock.sendto(str.encode(str(threshold)), serverAddressPort)
            except Exception as e:
                print(e)
                           
            # If 'c' pressed, calibrate
            key = cv2.waitKey(1)
            if key == ord('c'):
                
                preset, threshold = calibrate_threshold(hand_crop)
                print(f"[CALIBRATED] Preset: {preset}, Threshold: {threshold}")

    _, img_encoded = cv2.imencode('.jpg', frame)
    processed_data = img_encoded.tobytes()
    
    # Send the length of the processed frame and the frame itself
    client_socket.send(struct.pack('I', len(processed_data)))
    client_socket.send(processed_data)
    # Show info
    cv2.putText(frame, f"Preset: {preset}, Threshold: {threshold}", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'c' to calibrate with open hand", (10, 60), font, 0.6, (200, 200, 200), 1)
    if ShowDebugScreen:
        cv2.imshow("Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

client_socket.close()
sockImg.close()
cv2.destroyAllWindows()
