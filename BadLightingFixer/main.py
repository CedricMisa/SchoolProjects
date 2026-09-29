import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 400, 150)

cv2.createTrackbar("Clip Limit", "Settings", 3, 10, nothing)
cv2.createTrackbar("Grid Size", "Settings", 8, 32, nothing)

print("Starting Bad Lighting Fixer... Press 'q' to quit.")
print("Press 's' to save the enhanced AI-ready image.")

save_text_timer = 0
enhanced_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]
    
    clip_val = cv2.getTrackbarPos("Clip Limit", "Settings")
    clip_val = max(1.0, float(clip_val))
    
    grid_val = cv2.getTrackbarPos("Grid Size", "Settings")
    grid_val = max(4, grid_val if grid_val % 2 == 0 else grid_val + 1)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(grid_val, grid_val))
    
    enhanced_l = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    cv2.putText(frame, "RAW INPUT (Dark/Harsh Lighting)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(enhanced_image, "AI PRE-PROCESSED (CLAHE Enhanced)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if save_text_timer > 0:
        cv2.putText(enhanced_image, "ENHANCED IMAGE SAVED!", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        save_text_timer -= 1

    combined_dashboard = np.hstack((frame, enhanced_image))
    
    cv2.imshow("Bad Lighting Fixer Dashboard", combined_dashboard)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and enhanced_image is not None:
        cv2.imwrite("ai_enhanced_input.jpg", enhanced_image)
        save_text_timer = 30
        print("Enhanced image saved successfully!")

cap.release()
cv2.destroyAllWindows()
