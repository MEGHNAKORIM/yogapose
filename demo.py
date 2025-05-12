import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os


# Create a pose instance with improved detection parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    smooth_landmarks=True,
    static_image_mode=False
)

# Function to calculate angle between three points
def calculate_angle(landmark1, landmark2, landmark3,select=''):
    if select == '1':
        x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
        x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
        x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    else:
        radians = np.arctan2(landmark3[1] - landmark2[1], landmark3[0] - landmark2[0]) - np.arctan2(landmark1[1] - landmark2[1], landmark1[0] - landmark2[0])
        angle = np.abs(np.degrees(radians))
    
    angle_calc = angle + 360 if angle < 0 else angle
    #angle_calc = 360-angle_calc if angle_calc > 215 else angle_calc
    return angle_calc

# Initialize smoothing for angles
def smooth_angles(new_angle, prev_angle, alpha=0.8):
    if prev_angle is None:
        return new_angle
    return alpha * prev_angle + (1 - alpha) * new_angle

def correct_feedback(model, video=0, input_csv='0'):
    # Initialize previous angles for smoothing
    prev_angles = [None] * 12
    # Initialize pose classification counter
    pose_counter = 0
    last_pose = None
    # Load video
    cap = cv2.VideoCapture(video)  # Replace with your video path

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    accurate_angle_lists = []
    #Your accurate angle list
    # with open(input_csv, 'r') as inputCSV:
    #     for row in csv.reader(inputCSV):
    #         if row[12] == selectedPose: 
    #             accurate_angle_lists = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11])]

    angle_name_list = ["L-wrist","R-wrist","L-elbow", "R-elbow","L-shoulder", "R-shoulder", "L-knee", "R-knee","L-ankle","R-ankle","L-hip", "R-hip"]
    angle_coordinates = [[13, 15, 19], [14, 16, 18], [11, 13, 15], [12, 14, 16], [13, 11, 23], [14, 12, 24], [23, 25, 27], [24, 26, 28],[23,27,31],[24,28,32],[24,23,25],[23,24,26]]
    correction_value = 30

    fps_time = 0
   
    while cap.isOpened():
        ret_val, image = cap.read()
        #Resize the image to 50% of the original size
        # scale_factor = 1.5
        # image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        if not ret_val:
            break

        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_rgb = cv2.resize(image_rgb, (0, 0), None, .50, .50)
        # Get the pose landmarks
        results = pose.process(image_rgb)
        #save angle main
        angles = []
        
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            # Get the angle between the left elbow, wrist and left index points.
            left_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value],'1')
            angles.append(left_wrist_angle)
            # Get the angle between the right elbow, wrist and left index points.
            right_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value],'1')
            angles.append(right_wrist_angle)


            # Get the angle between the left shoulder, elbow and wrist points.
            left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],'1')
            angles.append(left_elbow_angle)
            # Get the angle between the right shoulder, elbow and wrist points.
            right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],'1')
            angles.append(right_elbow_angle)
            # Get the angle between the left elbow, shoulder and hip points.
            left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],'1')
            angles.append(left_shoulder_angle)

            # Get the angle between the right hip, shoulder and elbow points.
            right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],'1')
            angles.append(right_shoulder_angle)

            # Get the angle between the left hip, knee and ankle points.
            left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],'1')
            angles.append(left_knee_angle)

            # Get the angle between the right hip, knee and ankle points
            right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],'1')
            angles.append(right_knee_angle)

            # Get the angle between the left hip, ankle and LEFT_FOOT_INDEX points.
            left_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],'1')
            angles.append(left_ankle_angle)

            # Get the angle between the right hip, ankle and RIGHT_FOOT_INDEX points
            right_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value],'1')
            angles.append(right_ankle_angle)

            # Get the angle between the left knee, hip and right hip points.
            left_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],'1')
            angles.append(left_hip_angle)

            # Get the angle between the left hip, right hip and right kneee points
            right_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],'1')
            angles.append(right_hip_angle)
            
            y = model.predict([angles])
            
            Name_Yoga_Classification = str(y[0])

            # Get prediction probabilities for each class
            probabilities = model.predict_proba([angles])
            class_labels = model.classes_
            max_prob_idx = np.argmax(probabilities[0])
            max_prob = probabilities[0][max_prob_idx]
            predicted_pose = str(class_labels[max_prob_idx])
            
            # Implement pose stability check
            if max_prob > 0.7:  # Only accept high confidence predictions
                if predicted_pose == last_pose:
                    pose_counter += 1
                else:
                    pose_counter = 0
                last_pose = predicted_pose
                
                # Only update pose classification after stable detection
                if pose_counter > 5:  # Need 5 consistent frames
                    Name_Yoga_Classification = predicted_pose
                    check_accry_class = True
            else:
                check_accry_class = False

            with open(input_csv, 'r') as inputCSV:
                for row in csv.reader(inputCSV):
                    if row[12] == Name_Yoga_Classification: 
                        accurate_angle_lists = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11])]
                
            folder_path = 'F:/Anaconda_Project/Utilizing_Deep_Learning_for_Human_Pose_Estimation_in_Yoga/teacher_yoga/angle_teacher_yoga.csv'

            # Tiền tố cần kiểm tra
            prefix_to_match = Name_Yoga_Classification


            if check_accry_class == True :
                # # Duyệt qua tất cả các tệp trong thư mục
                # for file_name in os.listdir(folder_path):
                #     # Tạo đường dẫn đầy đủ của tệp
                #     file_path = os.path.join(folder_path, file_name)

                #     # Kiểm tra nếu tên tệp bắt đầu bằng tiền tố mong muốn
                #     if file_name.startswith(prefix_to_match):
                #         # Đọc và hiển thị ảnh (hoặc thực hiện các thao tác khác tùy ý)
                #         # image_ins = cv2.imread(file_path)
                #         # ins_resize = cv2.resize(image_ins, (0, 0), None, .50, .50)

                # Display the classification result in the bottom-left corner
                (w, h), _ = cv2.getTextSize(Name_Yoga_Classification, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(image, (10, image.shape[0] - 30), (10 + w, image.shape[0] - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, Name_Yoga_Classification, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            else :
                # Display the classification result in the bottom-left corner
                (w, h), _ = cv2.getTextSize('None', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(image, (10, image.shape[0] - 30), (10 + w, image.shape[0] - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, 'None', (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            
            correct_angle_count = 0
            for itr in range(12):
                point_a = (int(landmarks[angle_coordinates[itr][0]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][0]].y * image.shape[0]))
                point_b = (int(landmarks[angle_coordinates[itr][1]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][1]].y * image.shape[0]))
                point_c = (int(landmarks[angle_coordinates[itr][2]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][2]].y * image.shape[0]))

                # Calculate and smooth the angle
                raw_angle = calculate_angle(point_a, point_b, point_c, '0')
                angle_obtained = smooth_angles(raw_angle, prev_angles[itr])
                prev_angles[itr] = angle_obtained

                # Dynamic correction value based on the joint
                joint_correction = correction_value
                if 'shoulder' in angle_name_list[itr].lower() or 'hip' in angle_name_list[itr].lower():
                    joint_correction = correction_value * 1.2  # More tolerance for larger joints
                elif 'wrist' in angle_name_list[itr].lower() or 'ankle' in angle_name_list[itr].lower():
                    joint_correction = correction_value * 0.8  # Less tolerance for smaller joints

                # Calculate percentage of correctness
                target_angle = accurate_angle_lists[itr]
                diff_percentage = abs(angle_obtained - target_angle) / target_angle * 100

                if angle_obtained < target_angle - joint_correction:
                    status = f"Increase {angle_name_list[itr]} ({diff_percentage:.0f}% low)"
                elif angle_obtained > target_angle + joint_correction:
                    status = f"Decrease {angle_name_list[itr]} ({diff_percentage:.0f}% high)"
                else:
                    status = "Perfect!"
                    correct_angle_count += 1

                # Display status with color-coded feedback
                status_position = (point_b[0] - int(image.shape[1] * 0.15), point_b[1] + int(image.shape[0] * 0.03))
                status_color = (0, 255, 0) if status == "Perfect!" else (0, 165, 255)  # Green for perfect, orange for adjustment needed
                cv2.putText(image, f"{status}", status_position, cv2.FONT_HERSHEY_PLAIN, 1.2, status_color, 2)


                # Display angle values on the image
                cv2.putText(image, f"{angle_name_list[itr]}", (point_b[0] - 50, point_b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                
                # pos_onScreen = [200,1000]
                # # Chọn vị trí xuất hiện dựa trên giá trị của itr
                # cv2.putText(image, angle_name_list[itr]+":- %s" % (status), (pos_onScreen[itr%2], (itr+1)*60),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


                


            # Enhanced pose visualization
            mp_drawing = mp.solutions.drawing_utils
            # Custom drawing specs for better visibility
            custom_connections = mp_pose.POSE_CONNECTIONS
            landmark_drawing_spec = mp_drawing.DrawingSpec(
                color=(0, 255, 0),  # Green color for landmarks
                thickness=3,
                circle_radius=3
            )
            connection_drawing_spec = mp_drawing.DrawingSpec(
                color=(255, 255, 255),  # White color for connections
                thickness=2
            )
            # Draw the pose landmarks and connections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                custom_connections,
                landmark_drawing_spec,
                connection_drawing_spec
            )
            
            # Enhanced posture feedback with detailed guidance
            if correct_angle_count > 9:
                posture = "EXCELLENT FORM!"
                feedback = "Hold this position and breathe deeply"
                secondary_feedback = f"Maintaining {Name_Yoga_Classification} pose perfectly"
            elif correct_angle_count > 6:
                posture = "ALMOST THERE!"
                feedback = "Small adjustments needed"
                secondary_feedback = "Focus on the highlighted joints"
            else:
                posture = "ADJUSTING"
                feedback = "Follow the angle guides carefully"
                secondary_feedback = "Take your time to align properly"
            
            # Color coding for different states
            posture_color = {
                "EXCELLENT FORM!": (0, 255, 0),    # Green
                "ALMOST THERE!": (0, 165, 255),   # Orange
                "ADJUSTING": (0, 0, 255)          # Red
            }[posture]

            # Create a semi-transparent overlay for status display
            overlay = image.copy()
            # Draw background rectangle for better text visibility
            cv2.rectangle(overlay, (0, 0), (image.shape[1], 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Display enhanced feedback
            cv2.putText(image, f"Status: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2)
            cv2.putText(image, f"Guidance: {feedback}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, secondary_feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            # Display confidence percentage when pose is detected
            if check_accry_class:
                confidence_text = f"Confidence: {max_prob*100:.1f}%"
                cv2.putText(image, confidence_text, (image.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display FPS
            fps_text = f"FPS: {1.0 / (time.time() - fps_time):.1f}"
            cv2.putText(image, fps_text, (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
            # min_height = min(ins_resize.shape[0], resize_rgb.shape[0])

            # # Thay đổi kích thước ảnh để chiều dài bằng nhau
            # image_resized = cv2.resize(image, (int(min_height / image.shape[0] * image.shape[1]), min_height))
            # instruction_image_resized = cv2.resize(image_ins, (int(min_height / image_ins.shape[0] * image_ins.shape[1]), min_height))

            # # Ghép nối ảnh theo chiều ngang (50/50)
            # result = np.concatenate((image_resized, instruction_image_resized), axis=1)


            cv2.imshow('Mediapipe Pose Estimation', image)
            #cv2.imshow('Mediapipe Pose Estimation', result)
            fps_time = time.time()



            

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
