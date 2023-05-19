import cv2
import mediapipe as mp
import pyautogui

# initialize the camera, face mesh model, and get screen size
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    # capture a frame from the camera and flip it horizontally
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # convert the frame from BGR to RGB and feed it to the face mesh model for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    # get the height, width, and channels of the frame
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        # get the landmark points for the first face
        landmarks = landmark_points[0].landmark

        # draw a circle at the right eye landmark point
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            # move the mouse cursor to the right eye landmark point
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # get the landmark points for the left eye
        left = [landmarks[145], landmarks[159]]

        # draw a circle at each landmark point of the left eye
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # check if the left eye is closed
        if (left[0].y - left[1].y) < 0.004:
            # perform a left click
            pyautogui.click()
            pyautogui.sleep(1)

    # show the frame with the circles drawn on it
    cv2.imshow('Eye Controlled Mouse', frame)

    # wait for a key to be pressed
    cv2.waitKey(1)
