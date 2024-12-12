from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import mediapipe as mp

"""def draw_all_landmarks(image_shape, face_landmarks, background_image=None):
    if background_image is not None:
        img = np.copy(background_image)
        # Convert to RGBA if it's RGB
        if img.shape[2] == 3:
            img = np.dstack([img, np.ones(img.shape[:2], dtype=np.uint8) * 255])
    else:
        img = np.zeros((*image_shape[:2], 4))
    
    if not face_landmarks:
        return img
        
    for landmarks in face_landmarks:
        # Eyebrows (brown)
        eyebrow_points = {
            'left': [[276, 283], [283, 282], [282, 295], [295, 285], 
                    [285, 300], [300, 293], [293, 334], [334, 296]],
            'right': [[46, 53], [53, 52], [52, 65], [65, 55], 
                     [55, 70], [70, 63], [63, 105], [105, 66]]
        }
        for points in eyebrow_points.values():
            for p1_idx, p2_idx in points:
                pt1 = tuple(map(int, landmarks[p1_idx]))
                pt2 = tuple(map(int, landmarks[p2_idx]))
                cv2.line(img, pt1, pt2, (0.6, 0.3, 0.1, 1), 2)

        # Eyes (yellow)
        eye_points = [[33, 133], [133, 173], [173, 157], [157, 158], [158, 159],
                     [159, 160], [160, 33], [362, 263], [263, 466], [466, 388],
                     [388, 387], [387, 386], [386, 385], [385, 362]]
        for p1_idx, p2_idx in eye_points:
            pt1 = tuple(map(int, landmarks[p1_idx]))
            pt2 = tuple(map(int, landmarks[p2_idx]))
            cv2.line(img, pt1, pt2, (1, 1, 0, 1), 2)

        # Lips (cyan)
        lip_points = [[61, 146], [146, 91], [91, 181], [181, 84], [84, 17],
                     [17, 314], [314, 405], [405, 321], [321, 375], [375, 291],
                     [61, 185], [185, 40], [40, 39], [39, 37], [37, 0],
                     [0, 267], [267, 269], [269, 270], [270, 409]]
        for p1_idx, p2_idx in lip_points:
            pt1 = tuple(map(int, landmarks[p1_idx]))
            pt2 = tuple(map(int, landmarks[p2_idx]))
            cv2.line(img, pt1, pt2, (0, 1, 1, 1), 2)

        # Nose (magenta)
        nose_points = [[168, 6], [6, 197], [197, 195], [195, 5], [5, 4],
                      [4, 1], [1, 19], [19, 94], [94, 2], [2, 164]]
        for p1_idx, p2_idx in nose_points:
            pt1 = tuple(map(int, landmarks[p1_idx]))
            pt2 = tuple(map(int, landmarks[p2_idx]))
            cv2.line(img, pt1, pt2, (1, 0, 1, 1), 2)

        contour_points = [[356, 454],
                         [454, 323], [323, 361], [361, 288], [288, 397],
                         [397, 365], [365, 379], [379, 378], [378, 400],
                         [400, 377], [377, 152], [152, 148], [148, 176],
                         [176, 149], [149, 150], [150, 136], [136, 172],
                         [172, 58], [58, 132], [132, 93], [93, 234],
                         [234, 127]]

        for p1_idx, p2_idx in contour_points:
            pt1 = tuple(map(int, landmarks[p1_idx]))
            pt2 = tuple(map(int, landmarks[p2_idx]))
            cv2.line(img, pt1, pt2, (0.5, 0.5, 0.5, 1), 2)
    
    return img
"""
    

def draw_all_landmarks(image_shape, face_landmarks, background_image=None):
    if background_image is not None:
        img = np.copy(background_image)
    else:
        img = np.zeros(image_shape[:2], dtype=np.uint8)

    if not face_landmarks:
        return img
        
    for landmarks in face_landmarks:
        # Keep all original point definitions (eyebrow_points, eye_points, etc.)
        eyebrow_points = {
            'left': [[276, 283], [283, 282], [282, 295], [295, 285], 
                    [285, 300], [300, 293], [293, 334], [334, 296]],
            'right': [[46, 53], [53, 52], [52, 65], [65, 55], 
                     [55, 70], [70, 63], [63, 105], [105, 66]]
        }
        for points in eyebrow_points.values():
            for p1_idx, p2_idx in points:
                pt1 = tuple(map(int, landmarks[p1_idx]))
                pt2 = tuple(map(int, landmarks[p2_idx]))
                cv2.line(img, pt1, pt2, 0, 1, cv2.LINE_AA)

        eye_points = [[33, 133], [133, 173], [173, 157], [157, 158], [158, 159],
                     [159, 160], [160, 33], [362, 263], [263, 466], [466, 388],
                     [388, 387], [387, 386], [386, 385], [385, 362]]
        for p1_idx, p2_idx in eye_points:
            pt1 = tuple(map(int, landmarks[p1_idx]))
            pt2 = tuple(map(int, landmarks[p2_idx]))
            cv2.line(img, pt1, pt2, 0, 1, cv2.LINE_AA)

        lip_points = [[61, 146], [146, 91], [91, 181], [181, 84], [84, 17],
                     [17, 314], [314, 405], [405, 321], [321, 375], [375, 291],
                     [61, 185], [185, 40], [40, 39], [39, 37], [37, 0],
                     [0, 267], [267, 269], [269, 270], [270, 409]]
        for p1_idx, p2_idx in lip_points:
            pt1 = tuple(map(int, landmarks[p1_idx]))
            pt2 = tuple(map(int, landmarks[p2_idx]))
            cv2.line(img, pt1, pt2, 0, 1, cv2.LINE_AA)

        nose_points = [[168, 6], [6, 197], [197, 195], [195, 5], [5, 4],
                      [4, 1], [1, 19], [19, 94], [94, 2], [2, 164]]
        for p1_idx, p2_idx in nose_points:
            pt1 = tuple(map(int, landmarks[p1_idx]))
            pt2 = tuple(map(int, landmarks[p2_idx]))
            cv2.line(img, pt1, pt2, 0, 1, cv2.LINE_AA)

        contour_points = [[356, 454],
                         [454, 323], [323, 361], [361, 288], [288, 397],
                         [397, 365], [365, 379], [379, 378], [378, 400],
                         [400, 377], [377, 152], [152, 148], [148, 176],
                         [176, 149], [149, 150], [150, 136], [136, 172],
                         [172, 58], [58, 132], [132, 93], [93, 234],
                         [234, 127]]
        for p1_idx, p2_idx in contour_points:
            pt1 = tuple(map(int, landmarks[p1_idx]))
            pt2 = tuple(map(int, landmarks[p2_idx]))
            cv2.line(img, pt1, pt2, 0, 1, cv2.LINE_AA)
    
    return img


def get_face_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image)
        
    landmarks_arr = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmarks.append((x, y))
            landmarks_arr.append(np.array(landmarks))
    
    return landmarks_arr

def get_human_mask(img, conf=0.25):
    model = YOLO('yolov8m-seg.pt')
    results = model.predict(img, conf=conf)
    
    # Create black single-channel image
    img_copy = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Process YOLO detections
    for result in results:
        # Check if person class (id=0) is detected
        person_masks = [(mask.xy, box) for mask, box in zip(result.masks, result.boxes) if int(box.cls[0]) == 0]
        
        # Fill detected person masks with white
        for mask, _ in person_masks:
            points = np.int32([mask])
            cv2.fillPoly(img_copy, points, 255)
            
    return img_copy


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection)

"""def create_colored_segmentation(image_shape, masks):
    segmentation = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    
    colors = {
        'large': (150, 50, 50),    # Likely body/torso
        'medium': (50, 150, 50),   # Likely limbs/clothing
        'small': (50, 50, 150)     # Likely details/accessories
    }
    
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    total_area = sum(m['area'] for m in sorted_masks)
    
    for mask in sorted_masks:
        area_ratio = mask['area'] / total_area
        if area_ratio > 0.3:
            color = colors['large']
        elif area_ratio > 0.1:
            color = colors['medium']
        else:
            color = colors['small']
            
        segmentation[mask['segmentation']] = color
    
    return segmentation"""

def create_colored_segmentation(image_shape, masks):
    segmentation = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    total_area = image_shape[0] * image_shape[1]
    min_area = total_area / 100
    
    filtered_masks = [mask for mask in masks if mask['area'] > min_area]
    
    np.random.seed(42)
    colors = np.random.randint(50, 200, size=(len(filtered_masks), 3), dtype=np.uint8)
    
    for i, mask in enumerate(filtered_masks):
        segmentation[mask['segmentation']] = colors[i]
    
    return segmentation

def visualize_sam_masks(image, masks, alpha=0.5):
    result = image.copy()
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
    
    for idx, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        color = colors[idx]
        
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        
        cv2.addWeighted(colored_mask, alpha, result, 1-alpha, 0, result)
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color.tolist(), 2)
    
    return result  # Added return statement


def create_contour_drawing(image_shape, masks):
    contour_image = np.zeros(image_shape[:2], dtype=np.uint8)
    total_area = image_shape[0] * image_shape[1]
    min_area = total_area / 100

    filtered_masks = [mask for mask in masks if mask['area'] > min_area]

    for mask in filtered_masks:
        binary_mask = mask['segmentation'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(contour_image, [approx], -1, 255, 6, cv2.LINE_AA)

    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(contour_image, kernel, iterations=2)
    
    return 255 - eroded


def resize_image(image, max_dim=512):  # Changed from 1024 to 512
    height, width = image.shape[:2]
    scale = max_dim / float(max(height, width))
    if scale > 1:  # Keep this check even though it's unlikely with 512
        return image
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def main():
    #first checking if Cuda & Gpu Available
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "Not available")
    print("GPU device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    sam.to(device)  # Move model to GPU
    mask_generator = SamAutomaticMaskGenerator(sam)

    image = cv2.imread('./sample_images/eska2.webp')
    cv2.imshow("original image", image)
    cv2.waitKey(0)

    if image is None:
        print("Error: Could not read image")
        return
    
    image = resize_image(image)
    cv2.imshow("resized image", image)
    cv2.waitKey(0)

    human_mask = get_human_mask(image)
    cv2.imshow("human_mask ", human_mask )
    cv2.waitKey(0)

    if human_mask is None:
        print("No human detected in the image")
        return
    
    masked_image = image.copy()
    masked_image[human_mask<200] = [0, 0, 0]
    cv2.imshow("masked_image", masked_image)
    cv2.waitKey(0)

    masked_image_rgb=cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    sam_masks = mask_generator.generate(masked_image_rgb)
    print("done gather sam masks")

    contour_drawing = create_contour_drawing(masked_image.shape, sam_masks)
    cv2.imshow("contour_drawing", contour_drawing)
    cv2.waitKey(0)
    cv2.imwrite("contour_output.png", contour_drawing)
    
    segmented_parts = create_colored_segmentation(masked_image.shape, sam_masks)
    cv2.imshow("segmented_parts", segmented_parts)
    cv2.waitKey(0)

    face_landmarks = get_face_landmarks(image)
    landmarks_overlay = draw_all_landmarks(image.shape, face_landmarks, contour_drawing)
    cv2.imshow("landmarks_overlay",landmarks_overlay)
    cv2.waitKey(0)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(masked_image)
    plt.title('Segmented Human')
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.imshow(segmented_parts)
    plt.title('segmented parts')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(landmarks_overlay)
    plt.title('Parts + Landmarks')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('./output.jpeg', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()