import os
import face_recognition
import cv2

known_faces_dir = "known_faces"
unknown_faces_dir = "unknown_faces"
known_faces = []
known_names = []

# Load known faces and names
for name in os.listdir(known_faces_dir):
    known_names.append(name.replace(".jpg", ""))
    file = os.path.join(known_faces_dir, name)
    image = face_recognition.load_image_file(file)
    face_encoding = face_recognition.face_encodings(image)
    
    # Check if any face is found in the image
    if len(face_encoding) > 0:
        known_faces.append(face_encoding[0])
    else:
        print(f"No face found in the image: {file}")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Open a video capture feed (you can also use a webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Compare the found face to the known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        else:
            # Save the unknown face
            unknown_face_path = os.path.join(unknown_faces_dir, f"unknown_face_{len(os.listdir(unknown_faces_dir)) + 1}.jpg")
            cv2.imwrite(unknown_face_path, frame[top:bottom, left:right])

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name != "Unknown":
            color = (0, 255, 0)  # Green color for recognized faces
        else:
            color = (0, 0, 255)  # Red color for unrecognized faces

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
