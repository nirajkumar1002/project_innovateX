# project_innovateX
A major project to be created during summer school 2024 presented by CFI

This project contains 2 sections, 
Electronics and Software

To run the software part:
# To run the code properly, follow the following steps

step1. store the faces of your family members in the `known_faces` directory in the correct formate.
step2. run the `setup_database.py`
step3. run the `recognition.py`


# Work flow of the programmes

There are two python files 
    1. setup_database.py
    2. recognition.py

1. `setup_database.py`
    * there is directory `known_faces`, where we store the house members images
    * programme will detect this directory and iterate over each image
    * face of each image is detected then encoded (serialized),  and then both `name of image` and `encodings` are stored in a   face_encodings.pkl file
    * it will let you know if face is not detectable in any image.
    * for encoding/decoding of images, we have used pickle python library 
    * at last print a Success message if images are encodded and saved properly.

2. `recognition.py`
    * Open Web Cam and capture image of visitors and display the frame for a while
    * save the captured image in a new directory `unknown_faces`.
    * read the image from the unknown_faces directory, encode it and then save in `serialized_image.pkl` file.
    * now extract the encodings from the `face_encodings.pkl` file and match with the new encodings of visitor's image
    * used multiple images and different parameters for better accuracy for similar encoddings
    * if the distance between any two images is within the threshold value then it will Recognise the image with proper outputs.
    * finally it will 
        * return True, if image recognised
        * return False, if image not recognised
        * return False, if no face is detected in the image



# file structure for storing known_faces directory
1. known faces directory

known_faces/
    niraj/
        niraj1.jpg
        niraj2.jpg
        ..
        ..
        ..
    palak/
        palak1.jpg
        palak2.jpg
        ..
        ..

2. complete file directory 

security_1/
        known_faces/
        unknown_faces/
        face_encodings.pkl
        readme.md
        recognition.py
        serialized_image.pkl
        setup_database.py
        

