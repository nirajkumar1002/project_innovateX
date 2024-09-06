import cv2
import os
import face_recognition
import pickle
import numpy as np

# Directory to save captured unknown faces
unknown_faces_dir = 'unknown_faces'
# Path to the .pkl file containing known face encodings
database_path = 'face_encodings.pkl'

# Function to open webcam, capture image, and save it
def capture_save_unknown_face():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Display the captured frame (optional)
    cv2.imshow('Captured Image', frame)
    
    # Save the captured image to unknown_faces directory
    image_path = os.path.join(unknown_faces_dir, 'captured_unknown_face.jpg')
    cv2.imwrite(image_path, frame)
    
    # Wait for a key press and then close the OpenCV window
    cv2.waitKey(2000)  # Wait for 2 seconds (2000 milliseconds)
    cv2.destroyAllWindows()
    
    print(f"Captured unknown face saved: {image_path}")
    return image_path



# Function to serialize captured image using pickle
def serialize_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            with open('serialized_image.pkl', 'wb') as pkl_file:
                pickle.dump(image_data, pkl_file)
    except Exception as e:
        print(f"Error in serializing image: {e}")

# Function to match new encoding with encodings stored in .pkl file
def match_encoding_with_database(image_path, database_path, tolerance=0.29999): #0.29999
    try:
        # Load the captured image
        unknown_image = face_recognition.load_image_file(image_path)
        
        # Get face encoding of the captured image
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        
        if not unknown_encodings:
            return None, None, "No face found in the captured image."

        unknown_encoding = unknown_encodings[0]  # Assuming one face per image
        
        # Load known face encodings from .pkl file
        with open(database_path, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        
        # Compare the face encoding with known face encodings
        face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        
        # Find the best match based on the distance
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < tolerance:  # Threshold for considering a match
            confidence = 1 - face_distances[best_match_index]  # Higher confidence means lower distance
            return known_face_names[best_match_index], confidence, None
        
        return "Unknown Person", None, None
    
    except Exception as e:
        print(f"Error in matching encoding with database: {e}")
        return None, None, "Error in matching encoding with database."

# Main function to execute the process
def main():
    # Step 1: Capture and save unknown face
    image_path = capture_save_unknown_face()
    
    if image_path:
        # Step 2: Serialize captured image using pickle
        serialize_image(image_path)
        
        # Step 3: Match encoding with encodings stored in .pkl file
        matched_name, confidence, error_message = match_encoding_with_database(image_path, database_path)
        
        # Step 4: Print the matched name or error message
        if error_message:
            print(error_message)
            return False
        elif matched_name == "Unknown Person":
            print("No matching face found in the database.")
            return False
        else:
            print(f"Matched name: {matched_name}")
            print(f"Confidence Score: {confidence:.2f}")
            return True

if __name__ == "__main__":
    recognized = main()
    print(f"Person recognized: {recognized}")
