import cv2
import numpy as np

def nothing(x):
    pass

# 1. Initialize the webcam (0 is usually the built-in laptop camera)
cap = cv2.VideoCapture(0)

# 2. Create a settings window for live tuning during the presentation
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 400, 250)

# Trackbars let you adjust how sensitive the scratch detector is in real-time
cv2.createTrackbar("Blur Size", "Settings", 5, 20, nothing)
cv2.createTrackbar("Canny Min", "Settings", 30, 255, nothing)
cv2.createTrackbar("Canny Max", "Settings", 100, 255, nothing)
cv2.createTrackbar("Min Area", "Settings", 10, 500, nothing)

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # 3. Create an "Inspection Zone" in the middle of the screen
    # This prevents the camera from picking up background clutter like your face or desk
    height, width = frame.shape[:2]
    startX, startY = int(width*0.25), int(height*0.25)
    endX, endY = int(width*0.75), int(height*0.75)
    
    # Draw the blue inspection box on the screen
    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
    cv2.putText(frame, "INSPECTION ZONE", (startX, startY - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
    # Extract only the pixels inside the blue inspection box
    roi = frame[startY:endY, startX:endX]
    
    # 4. Convert the inspection area to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Get the current values from the trackbars
    blur_val = cv2.getTrackbarPos("Blur Size", "Settings")
    canny_min = cv2.getTrackbarPos("Canny Min", "Settings")
    canny_max = cv2.getTrackbarPos("Canny Max", "Settings")
    min_area = cv2.getTrackbarPos("Min Area", "Settings")
    
    # OpenCV requires the blur value to be an odd number greater than 0
    blur_val = blur_val if blur_val % 2 != 0 else blur_val + 1
    if blur_val < 1: blur_val = 1
    
    # 5. Apply Gaussian Blur to smooth out normal surface texture and camera grain
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    
    # 6. Apply Canny Edge Detection to find sharp lines (scratches)
    edges = cv2.Canny(blurred, canny_min, canny_max)
    
    # 7. Find Contours (grouping connected white pixels together)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defect_count = 0
    
    # 8. Loop through all detected edges and filter by size AND shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate aspect ratio (longer side divided by shorter side)
            # Add +1 to prevent division by zero errors
            longest_side = max(w, h)
            shortest_side = min(w, h) + 1 
            aspect_ratio = longest_side / shortest_side
            
            # A scratch is usually a line, meaning one side is much longer than the other.
            # If the aspect ratio is greater than 2.5, we count it as a scratch.
            if aspect_ratio > 2.5:
                defect_count += 1
                # Draw a red warning box around the scratch on the main frame
                cv2.rectangle(frame, (startX + x, startY + y), (startX + x + w, startY + y + h), (0, 0, 255), 2)

    # 9. Display the big PASS / FAIL text
    if defect_count == 0:
        cv2.putText(frame, "STATUS: PASS (Clean)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, f"STATUS: FAIL ({defect_count} scratches)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 10. Create a side-by-side dashboard
    # Convert the 1-channel grayscale edge map into a 3-channel BGR image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Create a blank black canvas the exact same size as the main camera frame
    edges_full_size = np.zeros_like(frame)
    # Paste the edges box directly into the middle of the black canvas
    edges_full_size[startY:endY, startX:endX] = edges_colored
    
    # Glue the main camera frame and the edge map together side-by-side
    combined_dashboard = np.hstack((frame, edges_full_size))
    
    # Show the single combined window
    cv2.imshow("Defect Detection Dashboard", combined_dashboard)
    
    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close windows when done
cap.release()
cv2.destroyAllWindows()
