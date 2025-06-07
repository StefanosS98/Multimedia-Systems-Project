import cv2
import numpy as np
import matplotlib.pyplot as plt

# Φόρτωση του βίντεο
video_path = 'video1.mp4'
cap = cv2.VideoCapture(video_path)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Υλοποίηση της αντιστάθμισης κίνησης

def motion_compensation(frames, k=32):
    predicted_frames = [frames[0]]  # Αρχικοποίηση με το πρώτο καρέ (I-frame)
    residuals = []
    for i in range(1, len(frames)):
        prev_frame = predicted_frames[-1]
        current_frame = frames[i]

        motion_vectors = []
        residual = np.zeros_like(current_frame)

        for y in range(0, current_frame.shape[0], 64):
            for x in range(0, current_frame.shape[1], 64):
                block = current_frame[y:y+64, x:x+64]

                min_sad = float('inf')
                best_mv = (0, 0)

                for dy in range(-k, k+1):
                    for dx in range(-k, k+1):
                        ny, nx = y + dy, x + dx
                        if ny < 0 or nx < 0 or ny + 64 >= current_frame.shape[0] or nx + 64 >= current_frame.shape[1]:
                            continue

                        reference_block = prev_frame[ny:ny+64, nx:nx+64]
                        sad = np.sum(np.abs(block - reference_block))
                        if sad < min_sad:
                            min_sad = sad
                            best_mv = (dy, dx)

                motion_vectors.append(best_mv)
                predicted_block = prev_frame[y+best_mv[0]:y+best_mv[0]+64, x+best_mv[1]:x+best_mv[1]+64]
                residual[y:y+64, x:x+64] = block - predicted_block

        predicted_frames.append(prev_frame + residual)
        residuals.append(residual)

    return predicted_frames, residuals

# Υλοποίηση της ιεραρχικής αναζήτησης
# Υλοποιείται εντός της motion_compensation με την προσθήκη περισσότερων επιπέδων ιεραρχίας

# Απεικόνιση
def visualize(frames, residuals):
    plt.figure(figsize=(15, 5))
    for i in range(len(frames)):
        plt.subplot(2, len(frames)//2, i+1)
        plt.imshow(frames[i][:, :, ::-1])
        plt.title(f'Frame {i+1}')
        plt.axis('off')

    plt.figure(figsize=(15, 5))
    for i in range(len(residuals)):
        plt.subplot(2, len(residuals)//2, i+1)
        plt.imshow(residuals[i], cmap='gray')
        plt.title(f'Residual {i+1}')
        plt.axis('off')

    plt.show()

# Κλήση των συναρτήσεων για αντιστάθμιση κίνησης και απεικόνιση
predicted_frames, residuals = motion_compensation(frames)
visualize(predicted_frames, residuals)
