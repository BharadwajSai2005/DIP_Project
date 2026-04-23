import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import winsound
import threading
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Alert sound
def play_alert():
    winsound.Beep(1000, 300)

cap = cv2.VideoCapture("Crowd-Activity-All.avi", cv2.CAP_FFMPEG)

ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (640, 360))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

prev_magnitude = None
frame_count = 0

mii_list = []
mii_history = []

threshold = 1.2
panic_count = 0

#  STATE MACHINE
NORMAL = 0
EARLY = 1
PANIC = 2
state = NORMAL

early_start = None
panic_start = None
last_beep_time = 0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total frames:", total_frames)
print("FPS:", fps)


#  GROUND TRUTH PANIC SEGMENTS
panic_segments = [
    (13, 19), (41, 47), (56, 63), (83, 89),
    (103, 109), (128, 132), (157, 163),
    (177, 183), (203, 207), (226, 230), (252, 256)
]

def is_ground_truth_panic(time_sec):
    for start, end in panic_segments:
        if (start - 3) <= time_sec <= (end + 3):   # ±2 sec tolerance
            return 1
    return 0

y_true = []
y_pred = []

# NEW: temporal smoothing buffer
prediction_buffer = []

def process_frame():
    global prev_gray, prev_magnitude, frame_count
    global mii_list, mii_history, threshold
    global state, early_start, panic_start, panic_count, last_beep_time
    global prediction_buffer

    ret, frame = cap.read()
    if not ret:
        return False

    frame_count += 1
    frame = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    

    #  OPTICAL FLOW
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 1, 10, 2, 3, 1.1, 0
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Heatmap overlay
    heatmap = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    frame = cv2.addWeighted(frame, 0.8, heatmap, 0.2, 0)

    #  MOTION FEATURES
    speed_variance = np.var(magnitude)

    hist, _ = np.histogram(angle, bins=8, range=(0, 2*np.pi), density=True)
    entropy = -np.sum(hist * np.log(hist + 1e-6))

    if prev_magnitude is not None:
        acceleration = np.mean(np.abs(magnitude - prev_magnitude))
    else:
        acceleration = 0

    prev_magnitude = magnitude

    #  MICRO-PATTERN GRID ANALYSIS
    h, w = gray.shape
    grid_h, grid_w = h // 3, w // 3
    panic_grids = 0   # important
    
    for i in range(3):
        for j in range(3):
            grid_mag = magnitude[i*grid_h:(i+1)*grid_h,j*grid_w:(j+1)*grid_w]

            if np.var(grid_mag) > 0.8*np.mean(magnitude):   # better threshold
                panic_grids += 1   # count grids
                cv2.rectangle(frame,
                              (j*grid_w, i*grid_h),
                              ((j+1)*grid_w, (i+1)*grid_h),
                              (0, 0, 255), 2)


    

    #  MII CALCULATION
    MII = (0.35 * speed_variance) + (0.25 * entropy) + (0.4 * acceleration)

    mii_list.append(MII)
    if len(mii_list) > 10:
        mii_list.pop(0)

    avg_mii = sum(mii_list) / len(mii_list)
    mii_history.append(avg_mii)

    #  IMPROVED AUTO THRESHOLD (smoothed)
    if len(mii_history) > 30:
        recent = mii_history[-30:]
        new_threshold = np.mean(recent) + 0.9 * np.std(recent)
        threshold = 0.8 * threshold + 0.2 * new_threshold  # NEW smoothing

    #  TREND SLOPE
    slope = 0
    if len(mii_history) > 12:
        recent = mii_history[-12:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]

    current_time = time.time()

    #  STATE MACHINE
    if state == NORMAL:
        if slope > 0.008:   # NEW: reduced noise
            if early_start is None:
                early_start = current_time
            elif current_time - early_start > 0.4:
                state = EARLY
        else:
            early_start = None

    elif state == EARLY:
        if avg_mii > threshold * 1.25 or panic_grids >= 3:
            if panic_start is None:
                panic_start = current_time
            elif current_time - panic_start > 0.35:
                state = PANIC
                panic_count += 1
                

                if current_time - last_beep_time > 2:
                    threading.Thread(target=play_alert).start()
                    last_beep_time = current_time
        elif slope <= 0:
            state = NORMAL

    elif state == PANIC:
        if avg_mii < threshold * 0.7:
            state = NORMAL
            early_start = None
            panic_start = None

    #  STATE DISPLAY
    if state == NORMAL:
        text = "Normal"
        color = (0, 255, 0)
    elif state == EARLY:
        text = "EARLY WARNING"
        color = (0, 165, 255)
    else:
        text = "STAMPEDE MAY HAPPEN(PANIC RISK)"
        color = (0, 0, 255)

    #  EVALUATION
    video_time = frame_count / fps   # NEW: correct timing
    gt = is_ground_truth_panic(video_time)
    

    # NEW: temporal smoothing prediction
    prediction_buffer.append(1 if state in (EARLY,PANIC) else 0)
    if len(prediction_buffer) > 6:
        prediction_buffer.pop(0)

    pred = 1 if sum(prediction_buffer) >= 2 else 0

    y_true.append(gt)
    y_pred.append(pred)

    #  TEXT OVERLAY
    cv2.putText(frame, f"MII: {avg_mii:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Threshold: {threshold:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, f"Slope: {slope:.4f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, f"Panic Count: {panic_count}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    
    cv2.putText(frame, f"Frame: {frame_count}", (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, f"Time: {video_time:.1f}s", (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Crowd Monitoring", frame)

    prev_gray = gray
    return True

#  MAIN LOOP
while True:
    result = process_frame()

    if result is False:
        print("Video ended")
        break

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()

#  FINAL METRICS
print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_true, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred))

print("\n===== ACCURACY =====")
print(accuracy_score(y_true, y_pred))

plt.plot(mii_history)
plt.title("MII Trend Over Time")
plt.xlabel("Frame")
plt.ylabel("MII")
plt.show()