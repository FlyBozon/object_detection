import os
import sys
import glob
import cv2
import numpy as np

def has_background(img):
    if img.shape[-1] == 4:
        alpha = img[..., 3]  
        non_transparent_pixels = np.count_nonzero(alpha)
        total_pixels = alpha.size
        
        transparency_ratio = non_transparent_pixels / total_pixels

        if transparency_ratio > 0.1:
            return True  
        else:
            return False  
    else:
        return True

def visualize_orientation_analysis(img, output_folder, base_title):
    vis_img = img.copy()
    if vis_img.shape[-1] == 4:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGRA2BGR)
    
    h, w = vis_img.shape[:2]
    
    gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY) if len(vis_img.shape) == 3 else vis_img.copy()
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    margin_percent = 0.1
    margin = int(h * margin_percent)
    
    top_y_end = h//3 + margin
    bottom_y_start = 2*h//3 - margin
    
    cv2.line(vis_img, (0, top_y_end), (w, top_y_end), (0, 255, 0), 2)
    cv2.line(vis_img, (0, bottom_y_start), (w, bottom_y_start), (0, 255, 0), 2)
    
    cv2.putText(vis_img, "Top Region", (10, top_y_end - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_img, "Bottom Region", (10, bottom_y_start + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    top_region = binary[:top_y_end]
    bottom_region = binary[bottom_y_start:]
    
    top_widths = []
    bottom_widths = []
    
    for y in range(top_region.shape[0]):
        row = top_region[y]
        if cv2.countNonZero(row) > 0:
            white_indices = np.where(row > 0)[0]
            row_width = white_indices[-1] - white_indices[0]
            top_widths.append(row_width)
            
            cv2.line(vis_img, (white_indices[0], y), (white_indices[-1], y), (0, 0, 255), 1)
    
    for y in range(bottom_region.shape[0]):
        row = bottom_region[y]
        if cv2.countNonZero(row) > 0:
            white_indices = np.where(row > 0)[0]
            row_width = white_indices[-1] - white_indices[0]
            bottom_widths.append(row_width)
            
            cv2.line(vis_img, (white_indices[0], y + bottom_y_start), 
                    (white_indices[-1], y + bottom_y_start), (255, 0, 0), 1)
    
    top_avg_width = np.mean(top_widths) if top_widths else 0
    bottom_avg_width = np.mean(bottom_widths) if bottom_widths else 0
    
    cv2.putText(vis_img, f"Top Avg Width: {top_avg_width:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis_img, f"Bottom Avg Width: {bottom_avg_width:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    handle_on_top = top_avg_width < bottom_avg_width
    decision_text = "Handle on Top" if handle_on_top else "Handle on Bottom"
    action_text = "Need to rotate 180°" if handle_on_top else "No additional rotation"
    
    cv2.putText(vis_img, decision_text, (10, h - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_img, action_text, (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    vis_path = os.path.join(output_folder, f"{base_title}_orientation_analysis.png")
    cv2.imwrite(vis_path, vis_img)
    
    return vis_img


def save_img(img, path, title):
    fragment_path = os.path.join(path, f'{title}.png')
    cv2.imwrite(fragment_path, img)

def remove_background_otsu(image):
    # Convert to BGR if image is BGRA
    if image.shape[-1] == 4:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr_image = image.copy()
    
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.dilate(mask, kernel, iterations=2)

    # Create a 4-channel output image (BGRA)
    bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask  # Set alpha channel to the mask

    return bgra, mask

def remove_background_adaptive(image):
    # Convert to BGR if image is BGRA
    if image.shape[-1] == 4:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr_image = image.copy()
    
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Create a 4-channel output image (BGRA)
    bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask  # Set alpha channel to the mask
    
    return bgra, mask

def remove_background_canny(image):
    # Convert to BGR if image is BGRA
    if image.shape[-1] == 4:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr_image = image.copy()
    
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1) #changed from 2
    
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Create a 4-channel output image (BGRA)
    bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask  # Set alpha channel to the mask
    
    return bgra, mask

def remove_background_combined(image):
    # Convert to BGR if image is BGRA
    if image.shape[-1] == 4:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr_image = image.copy()
    
    _, otsu_mask = remove_background_otsu(bgr_image)
    _, adaptive_mask = remove_background_adaptive(bgr_image)
    _, canny_mask = remove_background_canny(bgr_image)
    
    combined_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)
    combined_mask = cv2.bitwise_or(combined_mask, canny_mask)
    
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    #combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # Create a 4-channel output image (BGRA)
    bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = combined_mask  # Set alpha channel to the mask
    
    return bgra, combined_mask

def remove_background(image, method="combined"):
    if method == "otsu":
        result, _ = remove_background_otsu(image)
    elif method == "adaptive":
        result, _ = remove_background_adaptive(image)
    elif method == "canny":
        result, _ = remove_background_canny(image)
    elif method == "combined":
        result, _ = remove_background_combined(image)
    else:
        print(f"Unknown method {method}, falling back to combined")
        result, _ = remove_background_combined(image)
    
    # Check if there's any foreground detected
    if result.shape[-1] == 4:
        non_zero_pixels = cv2.countNonZero(result[:, :, 3])
    else:
        non_zero_pixels = cv2.countNonZero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        
    if non_zero_pixels == 0:
        return None
        
    return result

def load_images_from_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        sys.exit(1)

    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_paths:
        print("No images found in the folder.")
        sys.exit(1)

    return image_paths

def find_object_orientation(img):
    # Convert to grayscale
    if img.shape[-1] == 4:
        # Use alpha channel as mask for transparent images
        gray = img[:, :, 3].copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    rect = cv2.minAreaRect(largest_contour)
    center, (width, height), angle = rect
    
    is_vertical = height > width
    
    if is_vertical:
        rotation_angle = angle
    else:
        rotation_angle = angle + 90
    
    while rotation_angle > 90:
        rotation_angle -= 180
    while rotation_angle < -90:
        rotation_angle += 180
    
    return rotation_angle

def check_orientation_after_rotation(img):
    # Convert to grayscale
    if img.shape[-1] == 4:
        # Use alpha channel as mask for transparent images
        gray = img[:, :, 3].copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    h, w = binary.shape
    
    margin_percent = 0.1
    margin = int(h * margin_percent)
    
    top_region = binary[:h//3 + margin]
    bottom_region = binary[2*h//3 - margin:]
    
    top_pixels = cv2.countNonZero(top_region)
    bottom_pixels = cv2.countNonZero(bottom_region)
    
    top_width = 0
    bottom_width = 0
    top_rows_with_object = 0
    bottom_rows_with_object = 0
    
    for y in range(top_region.shape[0]):
        row = top_region[y]
        if cv2.countNonZero(row) > 0:
            white_indices = np.where(row > 0)[0]
            row_width = white_indices[-1] - white_indices[0]
            top_width += row_width
            top_rows_with_object += 1
    
    for y in range(bottom_region.shape[0]):
        row = bottom_region[y]
        if cv2.countNonZero(row) > 0:
            white_indices = np.where(row > 0)[0]
            row_width = white_indices[-1] - white_indices[0]
            bottom_width += row_width
            bottom_rows_with_object += 1
    
    avg_top_width = top_width / top_rows_with_object if top_rows_with_object > 0 else 0
    avg_bottom_width = bottom_width / bottom_rows_with_object if bottom_rows_with_object > 0 else 0
    
    need_rotation = avg_top_width < avg_bottom_width
    
    print(f"Top avg width: {avg_top_width:.2f}, Bottom avg width: {avg_bottom_width:.2f}")
    print(f"Need 180 degree rotation: {need_rotation}")
    
    return need_rotation

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Use INTER_CUBIC and BORDER_TRANSPARENT for transparent images
    if img.shape[-1] == 4:
        rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0, 0))  # Transparent border
    else:
        # Convert to BGRA first
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rotated = cv2.warpAffine(bgra, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0, 0))  # Transparent border
    
    return rotated

def crop_to_object(img):
    # Get alpha channel or convert to grayscale
    if img.shape[-1] == 4:
        gray = img[:, :, 3].copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img.shape[1], x_max + padding)
    y_max = min(img.shape[0], y_max + padding)
    
    cropped = img[y_min:y_max, x_min:x_max]
    
    return cropped

def standardize_size(img, target_size=(400, 600)):
    # Create a transparent canvas
    if img.shape[-1] == 4:
        standardized = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
    else:
        # Convert to BGRA first
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        standardized = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
    
    h, w = img.shape[:2]
    aspect = w / h
    target_aspect = target_size[0] / target_size[1]
    
    if aspect > target_aspect:
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect)
    
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    y_offset = (target_size[1] - new_height) // 2
    x_offset = (target_size[0] - new_width) // 2
    
    standardized[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return standardized

def save_all_stages(img, img_no_bg, img_oriented, img_cropped, img_standardized, output_folder, base_title):
    img_folder = os.path.join(output_folder, base_title)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    # For original image, keep it as is
    #save_img(img, img_folder, "1_original")
    
    # For all background removal methods, ensure we have transparent backgrounds
    otsu_result = remove_background(img, "otsu")
    if otsu_result is not None:
        #save_img(otsu_result, img_folder, "2a_otsu")
        pass
    
    adaptive_result = remove_background(img, "adaptive")
    if adaptive_result is not None:
        #save_img(adaptive_result, img_folder, "2b_adaptive")
        pass
    
    canny_result = remove_background(img, "canny")
    if canny_result is not None:
        pass
        #save_img(canny_result, img_folder, "2c_canny")
    
    #save_img(img_no_bg, img_folder, "2d_combined")
    
    visualize_orientation_analysis(img_no_bg, img_folder, base_title)
    
    #save_img(img_oriented, img_folder, "3_oriented")
    
    #save_img(img_cropped, img_folder, "4_cropped")
    
    #save_img(img_standardized, img_folder, "5_standardized")
    
    save_img(img_standardized, output_folder, base_title) #+ "_final")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py /path/to/input_folder /path/to/output_folder [bg_removal_method]")
        print("bg_removal_method options: otsu, adaptive, canny, combined (default)")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    bg_method = "combined"
    if len(sys.argv) > 3:
        bg_method = sys.argv[3]
        if bg_method not in ["otsu", "adaptive", "canny", "combined"]:
            print(f"Unknown background removal method: {bg_method}")
            print("Available options: otsu, adaptive, canny, combined")
            sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = load_images_from_folder(input_folder)
    
    target_size = (400, 600)

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        # Read image with transparency if it has it
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: failed to load {img_path}")
            continue

        # Make sure we're working with the right color space
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = 255  # Make fully opaque
        elif img.shape[-1] != 4:
            # Grayscale image, convert to BGRA
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            img[:, :, 3] = 255  # Make fully opaque

        if has_background(img):
            img_no_bg = remove_background(img, bg_method)
            if img_no_bg is None:
                print(f"No foreground detected in {img_path}, using original.")
                img_no_bg = img.copy()
        else: 
            img_no_bg = img.copy()

        angle = find_object_orientation(img_no_bg)
        print(f"Initial rotation by {angle:.2f} degrees to make vertical")
        img_vertical = rotate_image(img_no_bg, angle)
        
        need_flip = check_orientation_after_rotation(img_vertical)
        if need_flip:
            print("Rotating additional 180 degrees to put handle down")
            img_oriented = rotate_image(img_vertical, 180)
        else:
            img_oriented = img_vertical

        img_cropped = crop_to_object(img_oriented)

        img_standardized = standardize_size(img_cropped, target_size)

        base_title = os.path.splitext(os.path.basename(img_path))[0]
        #save_all_stages(img, img_no_bg, img_oriented, img_cropped, img_standardized, 
                       #output_folder, base_title)
        
        print(f"Processed and saved all stages for {base_title}")

        final_name = f"{base_title}"#_{bg_method}"
        save_img(img_standardized, output_folder, final_name)
        print(f"Saved final result: {os.path.join(output_folder, final_name + '.png')}")