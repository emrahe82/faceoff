from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import mediapipe as mp

def resize_image(image, max_dim=512):
    """Resize image maintaining aspect ratio so largest dimension is max_dim"""
    height, width = image.shape[:2]
    scale = max_dim / float(max(height, width))
    if scale > 1:
        return image
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def draw_face_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
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

def show_anns(anns, image, face_landmarks):
    if len(anns) == 0:
        return None
        
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    # Create a blank image for the boundaries
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    
    # Draw SAM segments
    for ann in sorted_anns:
        m = ann['segmentation']
        contours = cv2.findContours((m * 255).astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)[0]
        color = np.concatenate([np.random.random(3), [1]])
        cv2.drawContours(img, contours, -1, color, 2)
    
    # Draw face landmarks
    if face_landmarks:
        for landmarks in face_landmarks:
            # Eyebrows
            left_eyebrow_points = [[276, 283], [283, 282], [282, 295], [295, 285], [285, 300], [300, 293], [293, 334], [334, 296]]
            right_eyebrow_points = [[46, 53], [53, 52], [52, 65], [65, 55], [55, 70], [70, 63], [63, 105], [105, 66]]

            # Draw eyebrows (brown color)
            for points in [left_eyebrow_points, right_eyebrow_points]:
                for point_pair in points:
                    pt1 = landmarks[point_pair[0]]
                    pt2 = landmarks[point_pair[1]]
                    cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0.6, 0.3, 0.1, 1), 2)
            
            # Eyes (yellow)
            for eye_points in [[33, 133], [133, 173], [173, 157], [157, 158], [158, 159], [159, 160], [160, 33],  # Left eye
                             [362, 263], [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 362]]: # Right eye
                pt1 = landmarks[eye_points[0]]
                pt2 = landmarks[eye_points[1]]
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (1, 1, 0, 1), 2)
            
            # Lips (cyan)
            for lip_points in [[61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321], [321, 375], [375, 291], [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270], [270, 409]]:
                pt1 = landmarks[lip_points[0]]
                pt2 = landmarks[lip_points[1]]
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 1, 1, 1), 2)
            
            # Nose (magenta)
            for nose_points in [[168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19], [19, 94], [94, 2], [2, 164]]:
                pt1 = landmarks[nose_points[0]]
                pt2 = landmarks[nose_points[1]]
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (1, 0, 1, 1), 2)
            
            # Face contour (gray)
            for contour_points in [[10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10]]:
                pt1 = landmarks[contour_points[0]]
                pt2 = landmarks[contour_points[1]]
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0.5, 0.5, 0.5, 1), 2)

    return img  # Return the numpy array

def main():
    # Read image
    image = cv2.imread('./sample_images/man_)
    if image is None:
        print("Error: Could not read image")
        return
    
    # Resize image
    image = resize_image(image, max_dim=512)
    
    # Get face landmarks
    face_landmarks = draw_face_landmarks(image)
    
    # Convert to RGB for SAM
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize SAM
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    # Create side by side visualization
    plt.figure(figsize=(6,3))
    
    # Original image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Masks and landmarks on the right
    plt.subplot(1, 2, 2)
    mask_image = show_anns(masks, image, face_landmarks)
    if mask_image is not None:
        plt.imshow(mask_image)
        plt.title('Segments and Landmarks')
    plt.axis('off')

    # Save figure
    plt.savefig('./output.jpeg', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Found {len(masks)} segments in the image")
    print("Output saved as output.jpeg")

    #plt.tight_layout()
    #plt.show()
    
    print(f"Found {len(masks)} segments in the image")

if __name__ == "__main__":
    main()