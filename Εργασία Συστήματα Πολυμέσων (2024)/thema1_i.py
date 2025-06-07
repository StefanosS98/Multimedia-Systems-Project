import cv2
import numpy as np

# Διαβάζουμε το video
cap = cv2.VideoCapture('video1.mp4')

# Υποθέτουμε ότι το Frame 1 είναι I frame
ret, previous_frame = cap.read()
previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Αρχικοποιούμε τον κωδικοποιητή
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('reconstructed_video1.mp4', fourcc, 20.0, (previous_frame.shape[1], previous_frame.shape[0]), False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Μετατρέπουμε το πλαίσιο σε κλίμακα γκρι
    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Υπολογίζουμε το σφάλμα μεταξύ του τρέχοντος πλαισίου και του προηγούμενου
    error_frame = current_frame_gray - previous_frame_gray

    # Αποθηκεύουμε το πλαίσιο σφάλματος στο video αποκωδικοποίησης
    out.write(cv2.cvtColor(error_frame, cv2.COLOR_GRAY2BGR))

    # Το τρέχον πλαίσιο γίνεται το προηγούμενο για το επόμενο βήμα
    previous_frame_gray = current_frame_gray

# Απελευθερώνουμε τους πόρους
cap.release()
out.release()
cv2.destroyAllWindows()
