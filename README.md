# 🛡️ Smart Home Security System  
*A Summer School 2024 Project under CFI*  
**Team Members:** Niraj, Megha, Purab

---

## 🔧 Project Overview

This project is a prototype of a **Smart Home Security System**, integrating both **Electronics** and **Software** components. The software uses face recognition to identify known individuals and detect unknown visitors using a webcam.

---

## 🚀 Getting Started

To run the software module:

1. Store images of your family members inside the `known_faces/` directory, organized in subfolders (one per person).
2. Run `setup_database.py` to encode known faces.
3. Run `recognition.py` to perform real-time face recognition through webcam.

---

## 🔁 Workflow

### 1️⃣ `setup_database.py`

- Reads images from `known_faces/` directory.
- Detects and encodes each face using `face_recognition`.
- Saves encodings and corresponding names in `face_encodings.pkl` via `pickle`.
- Notifies if any face is undetectable.
- Prints a success message after storing all encodings.

### 2️⃣ `recognition.py`

- Captures visitor image via webcam and saves it in `unknown_faces/`.
- Encodes the captured image and stores it in `serialized_image.pkl`.
- Loads known encodings from `face_encodings.pkl` for comparison.
- Matches encodings based on distance threshold to detect known/unknown faces.
- Returns:
  - `True` → Face recognized
  - `False` → Face not recognized
  - `False` → No face detected

---

## 🗂️ Directory Structure

### 🔹 `known_faces/` Format

known_faces/ ├── niraj/ │ ├── niraj1.jpg │ ├── niraj2.jpg │ └── ... ├── palak/ │ ├── palak1.jpg │ ├── palak2.jpg │ └── ...


### 🔸 Full Project Structure

security_1/ ├── known_faces/ ├── unknown_faces/ ├── face_encodings.pkl ├── serialized_image.pkl ├── setup_database.py ├── recognition.py └── README.md


---

## 📝 Notes

- Ensure all face images have clear visibility and good lighting for accurate results.
- Encoding and recognition logic uses the `pickle` library for serialization of face data.
