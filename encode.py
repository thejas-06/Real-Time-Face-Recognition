import face_recognition
import os
import pickle

def load_and_encode_faces(dataset_path="dataset"):
    # This list will hold the 128 numbers (embedding) for each face
    known_face_encodings = []
    # This list will hold the person's name matching the embedding above
    known_face_names = []
    
    # We check if the dataset folder even exists
    if not os.path.exists(dataset_path):
        print(f"I couldn't find the '{dataset_path}' folder. Did you run collect.py?")
        return
        
    print("Starting face encoding process. This might take a minute...")
    
    # os.listdir will give us the folder names inside the dataset folder
    # Each folder name should be a person's name
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        # We only care about actual folders, not random files
        if not os.path.isdir(person_folder):
            continue
            
        print(f"Processing faces for: {person_name}")
        
        # Now let's loop through every photo inside this person's folder
        for image_filename in os.listdir(person_folder):
            # We only want to process image files (skip hidden system files)
            if not (image_filename.endswith(".jpg") or image_filename.endswith(".png")):
                continue
                
            image_path = os.path.join(person_folder, image_filename)
            
            # Load the image using face_recognition
            # It loads it in RGB format natively, which is exactly what we need!
            image = face_recognition.load_image_file(image_path)
            
            # Find the faces in the picture.
            # Even though our input images are already cropped faces, we still 
            # need to let the library locate the face to generate its embedding.
            locations = face_recognition.face_locations(image)
            
            # Now we extract the encodings (the 128D mathematical representation)
            encodings = face_recognition.face_encodings(image, locations)
            
            # We should only add it if a face encoding was actually created
            if len(encodings) > 0:
                # We just take the first face found in the image and save it
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
                
    # Now that we've processed all folders, let's bundle up the results
    # We use a dictionary so we can keep the encodings and names together
    data_to_save = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }
    
    # Pickle is a built-in Python way to save data structures directly to a file
    # We open our file in 'wb' mode which stands for "write binary"
    print("Saving all encodings to encodings.pkl...")
    with open("encodings.pkl", "wb") as file:
        pickle.dump(data_to_save, file)
        
    print("Encoding complete! You can now run recognize.py.")

if __name__ == "__main__":
    load_and_encode_faces()
