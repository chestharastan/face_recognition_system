import cv2
import os
import time

def capture_dataset():
    # 1. Load resources
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # 2. Setup directory
    save_dir = "face_dataset"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. Setup counters and timers
    count = 0
    max_images = 30
    last_capture_time = time.time()
    capture_interval = 1.0  # 1 second

    print(f"Starting capture... Look at the camera.")
    print(f"Collecting {max_images} images.")

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # Check if enough time has passed since the last capture
        current_time = time.time()
        
        # We only care about the first face detected for the dataset
        if len(faces) > 0:
            (x, y, w, h) = faces[0]

            # Logic: If 1 second has passed, save the image
            if current_time - last_capture_time >= capture_interval:
                
                # Crop the face (Region of Interest)
                face_roi = frame[y:y+h, x:x+w]
                
                # Create a unique filename
                img_name = f"{save_dir}/img_{count+1}.jpg"
                cv2.imwrite(img_name, face_roi)
                
                print(f"[Captured {count+1}/{max_images}] Saved to {img_name}")
                
                # Update counters
                count += 1
                last_capture_time = current_time
            
            # Draw rectangle for visualization (Green box)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display countdown/status on screen
            cv2.putText(frame, f"Count: {count}/{max_images}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        else:
            cv2.putText(frame, "No Face Detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Auto Capture Mode', frame)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Collection complete.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_dataset()