import cv2
import numpy as np

def nothing(x):
    pass

# 1. Initialize the webcam
cap = cv2.VideoCapture(0)

# 2. Create a settings window for live tuning during your presentation
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 400, 150)

# Trackbars to adjust CLAHE intensity live
cv2.createTrackbar("Clip Limit", "Settings", 3, 10, nothing) # Controls contrast amplification
cv2.createTrackbar("Grid Size", "Settings", 8, 32, nothing)  # Controls tile size for local lighting

print("Starting Bad Lighting Fixer... Press 'q' to quit.")
print("Press 's' to save the enhanced AI-ready image.")

save_text_timer = 0
enhanced_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Standardize frame size for a clean dashboard view
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]
    
    # Get current values from the trackbars
    clip_val = cv2.getTrackbarPos("Clip Limit", "Settings")
    clip_val = max(1.0, float(clip_val)) # Clip limit cannot be 0
    
    grid_val = cv2.getTrackbarPos("Grid Size", "Settings")
    grid_val = max(4, grid_val if grid_val % 2 == 0 else grid_val + 1) # Must be even and >= 4
    
    # 3. CLAHE Processing Pipeline
    # Step A: Convert the BGR image to LAB color space. 
    # L = Lightness (brightness), A & B = Color channels. This allows us to fix lighting without distorting colors!
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Step B: Initialize the CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(grid_val, grid_val))
    
    # Step C: Apply CLAHE exclusively to the Lightness channel
    enhanced_l = clahe.apply(l_channel)
    
    # Step D: Merge the enhanced lightness back with the original color channels
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    
    # Step E: Convert back to standard BGR color space for display
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    # 4. Add UI labels to both frames
    cv2.putText(frame, "RAW INPUT (Dark/Harsh Lighting)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(enhanced_image, "AI PRE-PROCESSED (CLAHE Enhanced)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display saved notification on the enhanced frame if 's' was pressed
    if save_text_timer > 0:
        cv2.putText(enhanced_image, "ENHANCED IMAGE SAVED!", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        save_text_timer -= 1

    # 5. Glue the raw camera feed and the enhanced AI feed together side-by-side
    combined_dashboard = np.hstack((frame, enhanced_image))
    
    cv2.imshow("Bad Lighting Fixer Dashboard", combined_dashboard)
    
    # 6. Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and enhanced_image is not None:
        cv2.imwrite("ai_enhanced_input.jpg", enhanced_image)
        save_text_timer = 30
        print("Enhanced image saved successfully!")

cap.release()
cv2.destroyAllWindows()
