import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load and encode known faces
bradpitt_image = face_recognition.load_image_file("brad_pitt.jpeg")
bradpitt_image_encoding = face_recognition.face_encodings(bradpitt_image)[0]

chico_image = face_recognition.load_image_file("chico.jpeg")
chico_image_encoding = face_recognition.face_encodings(chico_image)[0]

dicrapio_image = face_recognition.load_image_file("dicaprio.jpg")
dicrapio_image_encoding = face_recognition.face_encodings(dicrapio_image)[0]

elizabeth_image = face_recognition.load_image_file("elizabeth_olsen.jpeg")
elizabeth_image_encoding = face_recognition.face_encodings(elizabeth_image)[0]

tomhardy_image = face_recognition.load_image_file("tom_hardy.jpeg")
tomhardy_image_encoding = face_recognition.face_encodings(tomhardy_image)[0]

known_faces_encodings = [bradpitt_image_encoding, chico_image_encoding, dicrapio_image_encoding,
                         elizabeth_image_encoding, tomhardy_image_encoding]
known_faces_names = ["Brad Pitt", "Francisco Lachowski", "Leonardo DiCaprio", "Elizabeth Olsen", "Tom Hardy"]

# Prepare CSV for attendance
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")
with open(f"{current_date}.csv", "w+", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Name", "Time"])

    actors = known_faces_names.copy()  # Copy of known names to track attendance

    while True:
        # Capture a single frame from the video
        _, frame = video_capture.read()

        # Resize frame for faster processing and convert to RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and get encodings for each face in frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face in face_encodings:
            # Initialize name as "Unknown"
            name = "Unknown"

            # Compare with known faces and get distance
            matches = face_recognition.compare_faces(known_faces_encodings, face)
            face_distance = face_recognition.face_distance(known_faces_encodings, face)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            if name in actors:  # New recognition, log to CSV
                actors.remove(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                csv_writer.writerow([name, current_time])

            # Display the name on the video frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            linetype = 2
            cv2.putText(frame, name + " present", bottomLeftCornerText, font, fontScale, fontColor, thickness, linetype)

        # Display the frame
        cv2.imshow("Attendance", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
