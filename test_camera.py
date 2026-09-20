import cv2

print("================================")
print("   LAPTOP CAMERA TEST")
print("================================")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Laptop camera could not be opened.")
    print("Trying another camera index...")
    camera.release()

    camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("ERROR: No camera found.")
    exit()

print("SUCCESS: Laptop camera is working!")
print("Press Q to close the camera.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    cv2.imshow("Laptop Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Camera test finished.")