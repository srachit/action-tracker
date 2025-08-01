import mediapipe as mp
from mediapipe.tasks import python
import cv2 as cv


def draw_pose_landmarks_and_connections(frame, pose_landmarks, connections):
    """Draw both landmarks and connections."""
    height, width = frame.shape[:2]

    user_landmarks = set()
    for connection in connections:
        user_landmarks.add(connection[0])
        user_landmarks.add(connection[1])

    # Draw connections first (so they appear behind landmarks)
    for connection in connections:
        start_idx, end_idx = connection

        if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
            start_landmark = pose_landmarks[start_idx]
            end_landmark = pose_landmarks[end_idx]

            # Check visibility
            if (start_landmark.visibility > 0.5 and
                end_landmark.visibility > 0.5):
                start_x = int(start_landmark.x * width)
                start_y = int(start_landmark.y * height)
                end_x = int(end_landmark.x * width)
                end_y = int(end_landmark.y * height)

                # Draw connection line
                cv.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)

    # Draw landmarks on top
    for i, landmark in enumerate(pose_landmarks):
        if i in user_landmarks and landmark.visibility > 0.5:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv.circle(frame, (x, y), 10, (0, 255, 0), -1)

# Global variables for user selection and tracking
selected_people = []
current_frame_people = []
tracking_history = []  # Store recent positions for each tracked person

def mouse_callback(event, x, y, flags, param):
    global selected_people, current_frame_people

    if event == cv.EVENT_LBUTTONDOWN and param is not None:
        print(f"User clicked at pixel coordinates: ({x}, {y})")

        width = param['width']
        height = param['height']
        norm_x = x / width
        norm_y = y / height

        # Find closest person to click
        closest_person = find_closest_person_to_click(norm_x, norm_y, current_frame_people)
        if closest_person is not None:
            # Check if this person is already selected
            if closest_person not in selected_people:
                if len(selected_people) < 2:  # Only allow 2 selections
                    selected_people.append(closest_person)
                    print(f"Selected person {len(selected_people)}. Total selected: {len(selected_people)}")

                    if len(selected_people) == 2:
                        print("Both people selected! Press 'c' to continue tracking.")
                else:
                    print("Already selected 2 people. Press 'r' to reset selection.")
            else:
                print("This person is already selected!")

def find_closest_person_to_click(click_x, click_y, people_landmarks):
    """Find the person closest to the mouse click."""
    if not people_landmarks:
        return None

    min_distance = float('inf')
    closest_person = None

    for person_landmarks in people_landmarks:
        ref_point, ref_type = get_best_reference_point(person_landmarks)
        if ref_point:
            distance = ((click_x - ref_point.x)**2 + (click_y - ref_point.y)**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_person = person_landmarks

    return closest_person

def calculate_pose_similarity(person1, person2):
    """Calculate similarity between two poses using multiple landmarks."""
    # Use key landmarks for comparison (torso is most stable)
    key_landmarks = [0, 11, 12, 23, 24]  # nose, shoulders, hips

    total_distance = 0
    valid_comparisons = 0

    for idx in key_landmarks:
        if idx < len(person1) and idx < len(person2):
            lm1 = person1[idx]
            lm2 = person2[idx]

            # Only compare if both landmarks are reasonably visible
            if lm1.visibility > 0.3 and lm2.visibility > 0.3:
                distance = ((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)**0.5
                total_distance += distance
                valid_comparisons += 1

    return total_distance / valid_comparisons if valid_comparisons > 0 else float('inf')

def find_matching_person(selected_person, current_detections):
    """Find the best match for a selected person using improved tracking."""
    if not current_detections:
        return None

    min_similarity = float('inf')
    best_match = None

    for current_person in current_detections:
        # Calculate overall pose similarity
        similarity = calculate_pose_similarity(selected_person, current_person)

        if similarity < min_similarity:
            min_similarity = similarity
            best_match = current_person

    # Use stricter threshold to prevent jumping between people
    # Lower threshold = more strict matching
    return best_match if min_similarity < 0.15 else None


def get_center_obj(left_landmark, right_landmark):
    center_x = (left_landmark.x + right_landmark.x) / 2
    center_y = (left_landmark.y + right_landmark.y) / 2
    center_obj = type('obj', (object,), {
        'x': center_x,
        'y': center_y,
        'visibility': min(left_landmark.visibility, right_landmark.visibility)
    })
    return center_obj

def get_best_reference_point(person_landmarks):
    nose = person_landmarks[0]
    if nose.visibility > 0.7:
        return nose, "nose"

    left_shoulder = person_landmarks[11]
    right_shoulder = person_landmarks[12]
    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
        return get_center_obj(left_shoulder, right_shoulder), "shoulder"

    left_hip = person_landmarks[23]
    right_hip = person_landmarks[24]
    if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
        return get_center_obj(left_hip, right_hip), "hips"

    return None, "none"  # No good reference point found


def debug_landmark_visibility(frame, person_landmarks):
    """Draw visibility scores to understand which landmarks are reliable."""
    height, width = frame.shape[:2]

    # Check key landmarks
    key_landmarks = {
        0: "nose",
        11: "left_shoulder",
        12: "right_shoulder",
        23: "left_hip",
        24: "right_hip"
    }

    for idx, name in key_landmarks.items():
        landmark = person_landmarks[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        # Color based on visibility (green=good, red=poor)
        color = (0, int(255 * landmark.visibility), int(255 * (1 - landmark.visibility)))
        cv.circle(frame, (x, y), 8, color, -1)
        cv.putText(frame, f"{name}: {landmark.visibility:.2f}",
                   (x + 10, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def main():
    global current_frame_people, selected_people, tracking_history

    GIFT_CONNECTIONS = [
        (11, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]

    model_path = 'models/pose_landmarker_heavy.task'

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=3,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        video = cv.VideoCapture("resources/party.mp4")
        fps = video.get(cv.CAP_PROP_FPS)
        frame_num = 0

        # Set up mouse callback
        cv.namedWindow('Pose Tracking')
        cv.setMouseCallback('Pose Tracking', mouse_callback,
                           {'width': int(video.get(cv.CAP_PROP_FRAME_WIDTH)),
                            'height': int(video.get(cv.CAP_PROP_FRAME_HEIGHT))})

        print("Instructions:")
        print("1. Click on the gift receiver (person facing camera)")
        print("2. Click on the gift giver (person with back to camera)")
        print("3. Press 'c' to continue with tracking")
        print("4. Press 'q' to quit")

        selection_mode = True

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_num * int(1000 / fps))

            if pose_landmarker_result.pose_landmarks:
                current_frame_people = pose_landmarker_result.pose_landmarks

                if selection_mode:
                    # Draw all detected people for selection
                    for i, pose_landmarks in enumerate(pose_landmarker_result.pose_landmarks):
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for person 1, blue for person 2
                        draw_pose_landmarks_and_connections(frame, pose_landmarks, GIFT_CONNECTIONS)

                        # Draw selection indicators
                        if pose_landmarks in selected_people:
                            ref_point, ref_type = get_best_reference_point(pose_landmarks)
                            if ref_point:
                                height, width = frame.shape[:2]
                                x = int(ref_point.x * width)
                                y = int(ref_point.y * height)
                                cv.circle(frame, (x, y), 20, (0, 255, 255), 3)  # Yellow circle for selected
                                cv.putText(frame, f"SELECTED {len([p for p in selected_people if p == pose_landmarks])}",
                                         (x-50, y-30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Tracking mode - draw selected people with clear labels
                    tracked_count = 0
                    current_tracked = []

                    for i, selected_person in enumerate(selected_people):
                        # Use tracking history if available, otherwise use original selection
                        reference_person = tracking_history[i] if i < len(tracking_history) and tracking_history[i] else selected_person

                        # Find this person in current frame detections
                        matched_person = find_matching_person(reference_person, pose_landmarker_result.pose_landmarks)

                        if matched_person:
                            tracked_count += 1
                            current_tracked.append(matched_person)

                            # Draw with different colors for each tracked person
                            color = (0, 255, 0) if i == 0 else (255, 0, 255)  # Green for person 1, magenta for person 2
                            draw_pose_landmarks_and_connections(frame, matched_person, GIFT_CONNECTIONS)

                            # Add tracking labels
                            ref_point, ref_type = get_best_reference_point(matched_person)
                            if ref_point:
                                height, width = frame.shape[:2]
                                x = int(ref_point.x * width)
                                y = int(ref_point.y * height)

                                # Draw tracking indicator
                                label = f"PERSON {i+1} ({'RECEIVER' if i == 0 else 'GIVER'})"
                                cv.rectangle(frame, (x-60, y-40), (x+120, y-10), color, -1)
                                cv.putText(frame, label, (x-55, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv.circle(frame, (x, y), 15, color, 3)
                        else:
                            current_tracked.append(None)

                    # Update tracking history
                    tracking_history = current_tracked

                    # Show tracking status with confidence indicator
                    status_color = (0, 255, 0) if tracked_count == 2 else (0, 165, 255) if tracked_count == 1 else (0, 0, 255)
                    cv.putText(frame, f"TRACKING: {tracked_count}/2 people found",
                             (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            cv.imshow('Pose Tracking', frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(selected_people) >= 2:
                selection_mode = False
                print("Tracking mode activated!")
            elif key == ord('r'):
                selected_people = []
                selection_mode = True
                print("Selection reset. Click on people again.")

            frame_num += 1

    video.release()
    cv.destroyAllWindows()

main()
