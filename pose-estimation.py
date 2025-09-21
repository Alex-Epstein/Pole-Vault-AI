from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_angle(start, middle, end):
    vector1 = middle - start
    vector2 = end - middle
    dot_product = np.dot(vector1, vector2)
    magnitude_v1 = np.linalg.norm(vector1)
    magnitude_v2 = np.linalg.norm(vector2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def rate_takeoff(keypoints):
    # Measure knee angle to determine takeoff power
    knee_angle = compute_angle(keypoints[11], keypoints[13], keypoints[15])
    if knee_angle < 45:  # More bent knees mean better takeoff
        return 1.0
    elif knee_angle < 90:
        return 0.7
    return 0.5


def rate_invert(keypoints):
    # Measure if the body is fully inverted (torso angle)
    torso_angle = compute_angle(keypoints[11], keypoints[13], keypoints[15])
    if torso_angle < 90:  # Fully inverted
        return 1.0
    elif torso_angle < 120:
        return 0.7
    return 0.5


def rate_push_off(keypoints):
    # Measure arm angle during the push-off (elbow extension)
    elbow_angle = compute_angle(keypoints[5], keypoints[7], keypoints[9])
    if elbow_angle < 45:  # Elbows extended for better push-off
        return 1.0
    elif elbow_angle < 90:
        return 0.7
    return 0.5


class PoseEstimation:
    def __init__(self, video_name):
        self.model = YOLO('yolov8n-pose.pt')
        self.active_keypoints = [11, 13, 15, 5, 7, 9, 6, 8, 10, 12, 14]  # Add more keypoints (e.g., elbows, wrists)
        self.video_path = video_name
        self.scale = 1 / 2
        current_fps = 24
        desired_fps = 10
        self.skip_factor = current_fps // desired_fps
        self.center_x, self.center_y = 0, 0  # Default center position

    def analyze_pose(self, show_angle=False, confidence_threshold=0.5, tracking_area_threshold=0.4):
        if show_angle:
            plt.ion()
            fig, ax = plt.subplots()
            angle = 0
            angles = []
            time = []

        frame_count = 0
        color = (255, 255, 0)
        takeoff_score = 0
        invert_score = 0
        push_off_score = 0

        cv2.namedWindow("KeyPoints on Video", cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(self.video_path)

        prev_center = None  # To track previous bounding box center for motion tracking
        current_phase = "takeoff"  # Start with takeoff phase

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % self.skip_factor != 0:
                continue

            height, width, _ = frame.shape
            window_width = int(frame.shape[1] * self.scale)
            window_height = int(frame.shape[0] * self.scale)
            cv2.resizeWindow("KeyPoints on Video", window_width, window_height)

            # Run YOLO on the whole frame
            results = self.model(frame)

            largest_bbox = None
            largest_area = 0  # To find the largest detected person

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            largest_bbox = (x1, y1, x2, y2)

            if largest_bbox is not None:
                x1, y1, x2, y2 = largest_bbox
                # Calculate the center of the bounding box
                x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

                # Motion tracking: Compare the current and previous bounding box centers
                if prev_center is not None:
                    prev_x, prev_y = prev_center
                    movement_threshold = 50  # Adjust this threshold to account for the speed of movement
                    if abs(x_center - prev_x) > movement_threshold or abs(y_center - prev_y) > movement_threshold:
                        # This indicates that the person has moved significantly, likely the main subject
                        self.center_x, self.center_y = x_center, y_center
                prev_center = (x_center, y_center)

                # Ensure detection confidence is high enough before processing
                confidence = result.boxes.conf.cpu().numpy()[0]
                if confidence > confidence_threshold:
                    keypoints = result.keypoints.xy.cpu().numpy()[0]

                    # Rate different phases of the jump
                    takeoff_score = rate_takeoff(keypoints)
                    invert_score = rate_invert(keypoints)
                    push_off_score = rate_push_off(keypoints)

                    # Check for phase change (based on torso and limb angles)
                    if current_phase == "takeoff" and invert_score > 0.7:
                        current_phase = "invert"
                    elif current_phase == "invert" and push_off_score > 0.7:
                        current_phase = "push-off"

                    # **Draw Separate Lines for Each Limb and Torso**
                    # Draw Arms
                    cv2.line(frame, tuple(keypoints[5].astype(int)), tuple(keypoints[7].astype(int)), color, 8)  # Left arm
                    cv2.line(frame, tuple(keypoints[7].astype(int)), tuple(keypoints[9].astype(int)), color, 8)  # Left hand
                    cv2.line(frame, tuple(keypoints[6].astype(int)), tuple(keypoints[8].astype(int)), color, 8)  # Right arm
                    cv2.line(frame, tuple(keypoints[8].astype(int)), tuple(keypoints[10].astype(int)), color, 8)  # Right hand

                    # Draw Legs
                    cv2.line(frame, tuple(keypoints[11].astype(int)), tuple(keypoints[13].astype(int)), color, 8)  # Left leg
                    cv2.line(frame, tuple(keypoints[13].astype(int)), tuple(keypoints[15].astype(int)), color, 8)  # Left foot
                    cv2.line(frame, tuple(keypoints[12].astype(int)), tuple(keypoints[14].astype(int)), color, 8)  # Right leg
                    cv2.line(frame, tuple(keypoints[14].astype(int)), tuple(keypoints[16].astype(int)), color, 8)  # Right foot

                    # Draw Torso (spine, shoulders)
                    cv2.line(frame, tuple(keypoints[11].astype(int)), tuple(keypoints[12].astype(int)), color, 8)  # Spine line
                    cv2.line(frame, tuple(keypoints[11].astype(int)), tuple(keypoints[6].astype(int)), color, 8)  # Left shoulder
                    cv2.line(frame, tuple(keypoints[12].astype(int)), tuple(keypoints[6].astype(int)), color, 8)  # Right shoulder

                    if show_angle:
                        angle = compute_angle(keypoints[self.active_keypoints[0]], keypoints[self.active_keypoints[1]], keypoints[self.active_keypoints[2]])
                        cv2.putText(frame, f"{round(angle)}", (270, 1500), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 5, cv2.LINE_AA)

                        angles.append(angle)
                        time.append(frame_count)
                        ax.plot(time, angles, marker='o', color='orange')
                        plt.xlabel('Time')
                        plt.ylabel('Angle (degrees)')
                        plt.title('Angle vs. Time')
                        plt.draw()
                        plt.pause(.05)

            # Display the rating info on the video
            cv2.putText(frame, f"Takeoff: {takeoff_score*100:.2f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Invert: {invert_score*100:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Push-off: {push_off_score*100:.2f}%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Add final score and message
            total_score = takeoff_score + invert_score + push_off_score
            average_score = (total_score / 3) * 100
            message = "Great form!" if average_score > 80 else "Work on your technique"

            cv2.putText(frame, f"Total Score: {average_score:.2f}%", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Feedback: {message}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Show the frame with keypoints drawn
            cv2.imshow("KeyPoints on Video", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def run_analyze_pose(show_angle):
    pe = PoseEstimation('video3.mp4')  # Use the Mondo video file
    pe.analyze_pose(show_angle=show_angle, confidence_threshold=0.5, tracking_area_threshold=0.4)  # Adjust thresholds


if __name__ == '__main__':
    run_analyze_pose(show_angle=False)  # Run without showing angles
    # run_analyze_pose(show_angle=True)  # Uncomment this line to show the angles
