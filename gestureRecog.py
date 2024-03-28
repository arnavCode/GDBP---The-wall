import mediapipe as mp
import cv2
import numpy as np

# Create a GestureRecognizer instance with the live stream mode:
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_hands.min_detection_confidence = 0.7
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

#globals for tracking
leftTracking = False
rightTracking = False

#class for hand variables
class hand():
    def __init__(self, tracking, gesture, landmarkList):
        self.tracking = tracking
        self.gesture = gesture
        self.landmarkList = landmarkList
        if landmarkList != None:
            self.oldLoc = landmarkList[8]
        else:
            self.oldLoc = None

    #method to toggle tracking depending on gesture
    def checkTracking(self, gesture):
        if self.tracking == False and gesture == "Thumb_Up":
            self.tracking = True
        elif self.tracking == True and gesture == "Thumb_Down":
            self.tracking = False
        else:
            self.tracking = self.tracking
        return self.tracking

#globals for hands
leftHand = hand(leftTracking, None, None)
rightHand = hand(rightTracking, None, None)



def on_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global leftTracking, rightTracking, leftHand, rightHand

    #if hand detected
    if result.handedness:
        #if 2 hands detected
        if len(result.handedness) == 2:
            #find out which hand is which and assign
            if result.handedness[0][0].category_name == "Left":
                leftHand = hand(leftTracking, result.gestures[0][0].category_name, result.hand_landmarks[0])
                rightHand = hand(rightTracking, result.gestures[1][0].category_name, result.hand_landmarks[1])
            else:
                leftHand = hand(leftTracking, result.gestures[1][0].category_name, result.hand_landmarks[1])
                rightHand = hand(rightTracking, result.gestures[0][0].category_name, result.hand_landmarks[0])
            #check tracking and assign variable
            leftTracking = leftHand.checkTracking(leftHand.gesture)
            rightTracking = rightHand.checkTracking(rightHand.gesture)

        #if only one hand detected
        else:
            #assign hand and tracking
            if result.handedness[0][0].category_name == "Left":
                leftHand = hand(leftTracking, result.gestures[0][0].category_name, result.hand_landmarks[0])
                rightHand = hand(rightTracking, None, None)
                leftTracking = leftHand.checkTracking(leftHand.gesture)
            else:
                leftHand = hand(leftTracking, None, None)
                rightHand = hand(rightTracking, result.gestures[0][0].category_name, result.hand_landmarks[0])
                rightTracking = rightHand.checkTracking(rightHand.gesture)

def drawPointer(frame, hand):
    if hand.landmarkList != None: #double check if hand is detected, prevents a bug due to async processing of gesture result
        oldLoc = hand.oldLoc
        pointerLoc = hand.landmarkList[8]
        #small green circle at pointerLoc
        cv2.circle(frame, (int((pointerLoc.x)*frame.shape[1]), int(pointerLoc.y*frame.shape[0])), 30, (0,255,0), 1)
        #fainter green circle at oldLoc
        cv2.circle(frame, (int((oldLoc.x)*frame.shape[1]), int(oldLoc.y*frame.shape[0])), 30, (0,155,0), 1)
        oldLoc = pointerLoc
        return oldLoc

def drawGesture(frame, hand):
    #double check if hand is detected, prevents a bug due to async processing of gesture result
    if hand.landmarkList != None:
        #draw gesture name on frame
        cv2.putText(frame, hand.gesture, (10+int(frame.shape[1]*(1-hand.landmarkList[0].x)), int(hand.landmarkList[0].y*frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        return frame

def drawLandmarks(frame, hand):
    #double check if hand is detected, prevents a bug due to async processing of gesture result
    if hand.landmarkList != None:
        #draw landmarks on frame
        #mp_drawing.draw_landmarks(frame, hand.landmarkList, mp.solutions.hands.HAND_CONNECTIONS)
        print(hand.landmarkList)
        return frame


#gesture recognizer options
# Replace '/path/to/model.task' with the actual path to your model file
model_path = 'gesture_recognizer.task'
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback= on_result  # You can define your own result callback if needed
)



def main():
    # Create a GestureRecognizer instance with the live stream mode:
    with GestureRecognizer.create_from_options(options) as recognizer:
        #set drawing specs
        hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=5, color= (255,0,0))
        hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0,255,0))

        # Start capturing video from the webcam
        cap = cv2.VideoCapture(0)

        while cap.isOpened():



            global frame
            global leftTracking, rightTracking, leftHand, rightHand

            # Read the latest frame
            success, frame = cap.read()
            if not success:
                break


            # convert the frame to mp image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # get frame timestamp in ms
            frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            # Process the image with the GestureRecognizer
            recognizer.recognize_async(mp_image, frame_timestamp_ms)

            #if tracking is on for either hand, draw the pointer finger
            if leftTracking:
                leftHand.oldLoc = drawPointer(frame, leftHand)

            if rightTracking:
                rightHand.oldLoc = drawPointer(frame, rightHand)

            #flip frame so it's not a mirror view
            #flip before writing gesture name so text is not mirrored
            frame = cv2.flip(frame, 1)

            #have to use cheat to draw hand landmarks
            drawingResult = hands.process(frame)

            # if a gesture is detected, draw the gesture name
            if leftHand.gesture != None:
                drawGesture(frame, leftHand)
            if rightHand.gesture != None:
                drawGesture(frame, rightHand)


            if drawingResult.multi_hand_landmarks:
                for drawingLandmark in drawingResult.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=drawingLandmark,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_landmark_drawing_spec,
                        connection_drawing_spec=hand_connection_drawing_spec)


            # Visualize the results if needed
            cv2.imshow('MediaPipe Gesture Recognition', frame)

            # Press 'q' to quit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

main()