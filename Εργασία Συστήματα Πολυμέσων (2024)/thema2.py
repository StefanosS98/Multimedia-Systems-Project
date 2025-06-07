import cv2

# Διαβάζουμε το βίντεο
video_capture = cv2.VideoCapture('video2.mp4')

# Εξάγουμε τα πρώτα frames
_, first_frame = video_capture.read()
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

# Ορίζουμε το παράθυρο ROI (Region of Interest) για το αντικείμενο που θέλουμε να εξαφανίσουμε
# Σε αυτή την περίπτωση, το ορίζουμε χειροκίνητα, αλλά μπορείτε να χρησιμοποιήσετε έναν αλγόριθμο εντοπισμού αντικειμένων
# για να το κάνετε αυτόματα
roi = cv2.selectROI('Select Object to Remove and Press Enter', first_frame, False, False)

# Ορίζουμε τον αρχικό ολίσθησης
roi_x, roi_y, roi_w, roi_h = roi
roi_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Ορίζουμε τη μέθοδο ανίχνευσης
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Προετοιμάζουμε το βίντεο που θα αποθηκεύσουμε
output_video_path = 'output_video2.mp4'
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Αρχίζουμε την ανάγνωση του βίντεο
while True:
    _, frame = video_capture.read()
    if frame is None:
        break

    # Μετατροπή σε grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Υπολογισμός optical flow
    flow = cv2.calcOpticalFlowFarneback(previous_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Εφαρμογή αντιστάθμισης κίνησης στο frame
    dx = flow[..., 0]
    dy = flow[..., 1]
    for y in range(roi_y, roi_y + roi_h):
        for x in range(roi_x, roi_x + roi_w):
            new_y = int(y + dy[y, x])
            new_x = int(x + dx[y, x])
            if 0 <= new_y < frame.shape[0] and 0 <= new_x < frame.shape[1]:
                frame[y, x] = frame[new_y, new_x]


    # Εγγραφή frame στο νέο βίντεο
    output_video.write(frame)

    # Εμφάνιση του frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Ενημέρωση του προηγούμενου grayscale frame
    previous_gray = gray_frame.copy()

# Αποδέσμευση πόρων
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
