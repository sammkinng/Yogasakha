from mediapipe import solutions
from cv2 import COLOR_BGR2RGB,cvtColor
from utils.utils import calculate_angle, is_same_side_of_line,are_points_collinear,determine_orientation

# # Initialize MediaPipe Pose
mp_pose = solutions.pose
pose = mp_pose.Pose()

def test_pose(pos,image):
    image_rgb = cvtColor(image, COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
   
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        left_elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y)
        left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y)

        right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        right_elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y)

        left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
        left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
       
        right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
        right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
        if pos==1:
            if (abs(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x-landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x)<=0.02): #0.005 ->0.01
                return True ,None
                  
            return False,None
        if pos==2:
            
            if (are_points_collinear(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])):
                
                if(not is_same_side_of_line(landmarks[mp_pose.PoseLandmark.RIGHT_EAR],landmarks[mp_pose.PoseLandmark.RIGHT_HEEL],landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])):
                    return True,None
                return False,1
            return False,2
        if pos==3:
            
            if (abs(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y-landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y)<=0.01):
                return True,None
      
            return False,None
        if pos==4:
            if (-landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y+landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y or
                -landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y+landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y):
                return True,None
            return False,None
        if pos==5:
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            if ((left_hip_angle>=170 and left_hip_angle<=180 ) or (right_hip_angle>=170 and right_hip_angle<=180)):
                return True,None
            return False,None
        if pos==6:
          
            if (are_points_collinear(landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT],landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],tolerance=.01035)):
               
                if(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y):
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                   
                    if((left_elbow_angle>=80 and left_elbow_angle<=100)or(right_elbow_angle>=80 and right_elbow_angle<=100)):
                        return True,None
                 
                    return False,1 
                
                return False,2
          
            return False,3
        if pos==7:
            return True,None
        if pos==8:
            return True,None
    return False

def extract_features(image):
    image_rgb = cvtColor(image, COLOR_BGR2RGB)
    results = pose.process(image_rgb)
   
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates of key joints
        left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        left_elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y)
        left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y)

        right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        right_elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y)

        left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
        left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
        left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)

        right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
        right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
        right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

        # Calculate angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_shoulder)
        right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_shoulder)
        left_ankle_angle = calculate_angle(left_knee, left_ankle, left_hip)
        right_ankle_angle = calculate_angle(right_knee, right_ankle, right_hip)

        # Determine orientation
        orientation = determine_orientation(left_shoulder, left_hip)
       
        return [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle,
                        left_hip_angle, right_hip_angle, left_shoulder_angle, right_shoulder_angle,
                        left_wrist_angle, right_wrist_angle, left_ankle_angle, right_ankle_angle,
                        orientation]
