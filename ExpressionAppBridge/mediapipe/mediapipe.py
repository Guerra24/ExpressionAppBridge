from sys import platform

if platform == "linux":
  import cv2
else:
  from ExpressionAppBridge.mediapipe.camera import create_camera_backend

from ExpressionAppBridge.tracking_data import TrackingData

import time, transforms3d, threading

# Math constants
ROT_X_FACTOR = 80
ROT_Y_FACTOR = -100
ROT_Z_FACTOR = -60
POS_X_FACTOR = -0.01
POS_Y_FACTOR = 0.01
POS_Z_FACTOR = 0.01
SYNC_EYE_BLINK = True

# Mapping from mediapipe parameters to iFM
# The mediapipe parameters seem to be mirrored so Left and Right are mapped opposite
mediapipe_to_ifm = {
    "browInnerUp": "browInnerUp",
    "browDownLeft": "browDown_R",
    "browDownRight": "browDown_L",
    "browOuterUpLeft": "browOuterUp_R",
    "browOuterUpRight": "browOuterUp_L",
    # Eye
    "eyeLookUpLeft": "eyeLookUp_R",
    "eyeLookUpRight": "eyeLookUp_L",
    "eyeLookDownLeft": "eyeLookDown_R",
    "eyeLookDownRight": "eyeLookDown_L",
    "eyeLookInLeft": "eyeLookIn_R",
    "eyeLookInRight": "eyeLookIn_L",
    "eyeLookOutLeft": "eyeLookOut_R",
    "eyeLookOutRight": "eyeLookOut_L",
    "eyeBlinkLeft": "eyeBlink_R",
    "eyeBlinkRight": "eyeBlink_L",
    "eyeSquintLeft": "eyeSquint_R",
    "eyeSquintRight": "eyeSquint_L",
    "eyeWideLeft": "eyeWide_R",
    "eyeWideRight": "eyeWide_L",
    # Cheek
    "cheekPuff": "cheekPuff",
    "cheekSquintLeft": "cheekSquint_R",
    "cheekSquintRight": "cheekSquint_L",
    # Nose
    "noseSneerLeft": "noseSneer_R",
    "noseSneerRight": "noseSneer_L",
    # Jaw
    "jawOpen": "jawOpen",
    "jawForward": "jawForward",
    "jawLeft": "jawRight",
    "jawRight": "jawLeft",
    # Mouth
    "mouthFunnel": "mouthFunnel",
    "mouthPucker": "mouthPucker",
    "mouthLeft": "mouthRight",
    "mouthRight": "mouthLeft",
    "mouthRollUpper": "mouthRollUpper",
    "mouthRollLower": "mouthRollLower",
    "mouthShrugUpper": "mouthShrugUpper",
    "mouthShrugLower": "mouthShrugLower",
    "mouthClose": "mouthClose",
    "mouthSmileLeft": "mouthSmile_R",
    "mouthSmileRight": "mouthSmile_L",
    "mouthFrownLeft": "mouthFrown_R",
    "mouthFrownRight": "mouthFrown_L",
    "mouthDimpleLeft": "mouthDimple_R",
    "mouthDimpleRight": "mouthDimple_L",
    "mouthUpperUpLeft": "mouthUpperUp_R",
    "mouthUpperUpRight": "mouthUpperUp_L",
    "mouthLowerDownLeft": "mouthLowerDown_R",
    "mouthLowerDownRight": "mouthLowerDown_L",
    "mouthPressLeft": "mouthPress_R",
    "mouthPressRight": "mouthPress_L",
    "mouthStretchLeft": "mouthStretch_R",
    "mouthStretchRight": "mouthStretch_L",
    "tongueOut": "tongueOut"
}

def process_BlendShapes_into_TrackingData(Blendshapes, tracking_data):

    eyeBlinkAverage = 0.0
    for C in Blendshapes[0]:
        if C.category_name in mediapipe_to_ifm.keys():
            if SYNC_EYE_BLINK and (C.category_name == 'eyeBlinkLeft' or C.category_name == 'eyeBlinkRight'):
                eyeBlinkAverage += C.score
            else:
                tracking_data.blendshapes[mediapipe_to_ifm[C.category_name]] = C.score * 100
    if SYNC_EYE_BLINK:
        eyeBlinkAverage /= 2.0
        eyeBlinkAverage *= 100
        # print(eyeBlinkAverage, end='\r')
        tracking_data.blendshapes[mediapipe_to_ifm['eyeBlinkLeft']] = eyeBlinkAverage
        tracking_data.blendshapes[mediapipe_to_ifm['eyeBlinkRight']] = eyeBlinkAverage

def mediapipe_start(cal, iFM, camera, camera_cap):
    
    # Import mediapipe
    import mediapipe as mp
    
    # Running Constants
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Temporary tracking data storage
    temp_td = TrackingData()
    # mediapipe does not give us confidence
    temp_td.confidence = 100
    
    # FaceLandmarker callback function
    def onDetect(DetectionResult, Image, Cnf):
        try:
            affine = transforms3d.affines.decompose(DetectionResult.facial_transformation_matrixes[0])
            translation = affine[0]
            rotation_euler = transforms3d.euler.mat2euler(affine[1])
            
            temp_td.head[0] = rotation_euler[0] * ROT_X_FACTOR
            temp_td.head[1] = rotation_euler[1] * ROT_Y_FACTOR
            temp_td.head[2] = rotation_euler[2] * ROT_Z_FACTOR
            
            temp_td.head[3] = translation[0] * POS_X_FACTOR
            temp_td.head[4] = translation[1] * POS_Y_FACTOR
            temp_td.head[5] = translation[2] * POS_Z_FACTOR
            
            process_BlendShapes_into_TrackingData(DetectionResult.face_blendshapes, temp_td)
            
            cal.input_tracking(temp_td)
            
            # payload = str(iFM)
            iFM.udp_send()
            # print(f"Running... {int(1/(time.time() - start))} FPS", end='\r')
        except IndexError:
            pass
    
    # Set landmarker options
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        result_callback=onDetect)
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        # Create camera backend
        if platform == "linux":
            print("Using OpenCV for camera")
            cap = cv2.VideoCapture(camera)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(width, height)
        else:
            cap = create_camera_backend(camera, camera_cap)
        
        start = None
        _, frame = cap.read()
        try:
            while True:
                # start = time.time()
                # Capture frame-by-frame
                ret, newframe = cap.read(frame)
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Convert to mediapipe format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=newframe)
                
                # Send to landmarker, timestamp in ms
                landmarker.detect_async(mp_image, int(time.perf_counter() * 1000))
        except KeyboardInterrupt:
            print("Closing...")