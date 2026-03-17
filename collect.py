import cv2
import os
import face_recognition

def setup_directory(person_name):
    # We want to create a folder for the person to save their face images.
    # If the folder doesn't exist, this will make it for us.
    save_path = f"dataset/{person_name}"
    os.makedirs(save_path, exist_ok=True)
    return save_path

def collect_faces():
    # First, let's ask for the name of the person we are capturing
    person_name = input("Enter the name of the person: ").strip()
    
    # Check if they actually typed something 
    if not person_name:
        print("Name cannot be empty! Run the script again.")
        return

    # Let's get our folder ready
    save_folder = setup_directory(person_name)
    
    # Time to turn on the webcam (0 means the default laptop camera)
    print("Starting webcam... Look at the camera!")
    video_capture = cv2.VideoCapture(0)
    
    photos_captured = 0
    max_photos = 30
    
    # We will keep looping and reading frames until we have exactly 30 photos
    while photos_captured < max_photos:
        # Read a single frame from the camera
        # 'ret' is just a boolean saying if it was successful or not
        ret, frame = video_capture.read()
        
        if not ret:
            print("Failed to grab frame. Check your webcam!")
            break

        # face_recognition library prefers RGB (Red-Green-Blue), 
        # but OpenCV reads the exact opposite: BGR (Blue-Green-Red).
        # We need to convert the color space so face_recognition understands it.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Let's find all the faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # For simplicity, if there's at least one face, let's focus on the first one
        if len(face_locations) > 0:
            # The location gives us four borders of the face
            top, right, bottom, left = face_locations[0]
            
            # We want to save *only* the face, dropping the noisy background.
            # Array slicing in python is awesome for cropping images!
            face_image = frame[top:bottom, left:right]
            
            # Now let's save this cropped face image to our folder
            # Naming it 1.jpg, 2.jpg, 3.jpg, etc.
            photos_captured += 1
            file_name = f"{save_folder}/{photos_captured}.jpg"
            cv2.imwrite(file_name, face_image)
            
            # Provide some visual feedback to the user so they know we are capturing
            # Let's draw a nice green bounding box around their face on the screen
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            print(f"Captured {photos_captured}/{max_photos} photos...")

        # Let's show the live video feed to the user!
        cv2.imshow('Face Collection (Press Q to quit)', frame)
        
        # This waits 1 millisecond for the user to press a key. 
        # It's also required for cv2.imshow to update the window properly
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting early!")
            break

    # Once we have 30 photos or the user quit, clean everything up
    print("Done collecting faces! Cleaning up camera resources...")
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_faces()
