import cv2
import face_recognition
import pickle

def start_recognition():
    # First, let's try to load our saved encodings
    # If the file isn't there, we can't recognize anyone!
    try:
        with open("encodings.pkl", "rb") as file:
            data = pickle.load(file)
            known_encodings = data["encodings"]
            known_names = data["names"]
    except FileNotFoundError:
        print("I couldn't find encodings.pkl. Did you run encode.py first?")
        return
        
    print("Encodings loaded successfully! Starting the webcam...")
    
    # 0 is the default webcam on most laptops
    video_capture = cv2.VideoCapture(0)
    
    # Start our infinite loop for our real-time video feed
    while True:
        # Read the current frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting.")
            break
            
        # OpenCV uses BGR coloring, but face_recognition needs RGB
        # We must convert it before finding or encoding faces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # We process each face together with its location in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            # By default, we assume we don't know who this is
            name = "Unknown"
            text_color = (0, 0, 255) # Red color in BGR for unknown matches
            
            # We only try to match if we actually have some known people mapped out
            if len(known_encodings) > 0:
                # face_distance compares our current face to all known faces
                # It returns an array of distance numbers. Smaller numbers mean a better match
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                # Convert the numpy array to a simple python list 
                # to easily find the minimum distance and its index
                distances_list = face_distances.tolist()
                min_distance = min(distances_list)
                best_match_index = distances_list.index(min_distance)
                
                # We use a strict threshold of 0.45 just like the instructions asked
                # Anything below 0.45 means we are confident it's the exact same person
                if min_distance < 0.45:
                    name = known_names[best_match_index]
                    text_color = (0, 255, 0) # Green color in BGR for known match
            
            # Draw a green bounding box around every face it detects on camera
            box_color = (0, 255, 0) # Green box
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Let's put the name right above the bounding box
            # We leave a small 10-pixel gap upwards so it doesn't overlap the line
            cv2.putText(frame, name, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

        # Let's finally show the beautiful frame to the user with all tags and boxes
        cv2.imshow("Face Recognition LiveFeed (Press Q to quit)", frame)
        
        # Wait 1 millisecond. If 'q' is pressed, break the video loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video stream ended by user.")
            break

    # We always want to be polite and release the camera when done!
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_recognition()
