import face_recognition
import os
import pickle

# Path to the directory with known faces
known_faces_dir = 'known_faces'
database_path = 'face_encodings.pkl'

# Initialize lists to hold face encodings and names
known_face_encodings = []
known_face_names = []
errors = []

# Process each subdirectory (each representing a person) in the directory
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    
    # Check if the path is a directory
    if os.path.isdir(person_dir):
        for file_name in os.listdir(person_dir):
            # Load the image file
            image_path = os.path.join(person_dir, file_name)
            print(file_name)
            image = face_recognition.load_image_file(image_path)
            
            # Detect face(s) in the image and get the face encoding(s)
            face_encodings = face_recognition.face_encodings(image)
            print(face_encodings)
            
            if face_encodings:
                # We assume each image contains one face
                face_encoding = face_encodings[0]
                
                # Store the face encoding and the name (assumed from the directory name)
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)
            else:
                errors.append(f"No face detected in image: {file_name}")

# Save the encodings and names to a file
with open(database_path, 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

# Print errors (if any)
if errors:
    print("\nErrors:")
    for error in errors:
        print(error)

print("\nKnown faces loaded and encoded successfully.")
