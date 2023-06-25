import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.cm as cm
import yaml

config = yaml.safe_load(open("C:/Users/fawaz/Documents/Github/WhereAmongUs/configuration.yaml"))

'''
IOU (Intersection over Union) is a metric used to evaluate the performance of a segmentation algorithm.
It is calculated as the ratio of the area of intersection between the predicted segmentation mask and the ground truth segmentation mask
to the area of union between the predicted segmentation mask and the ground truth segmentation mask.
'''
def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


'''
This funciton takes in a path and reads the image in as a mask along with resizing it.
'''
def read_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask / 255

'''
Crop mask takes in the inputted mask and crops it to the smallest possible size
based upon the first pixels to each side of the image.
'''
def crop_mask(mask):
    row_indices = np.where(np.any(mask == 1, axis=1))[0]
    col_indices = np.where(np.any(mask == 1, axis=0))[0]

    top = row_indices.min()
    bottom = row_indices.max()
    left = col_indices.min()
    right = col_indices.max()

    return mask[top:bottom+1, left:right+1]



def generate_output_image(input_image, masks_with_scores, overlay_image_path, output_path):
    
    #Alpha refers to the alpha of the mask to be applied to the masks of the image.
    alpha = 0.8

    #Overlay will hold the original image with the masks applied to it.
    overlay = input_image.copy()
    overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

    #Cycle through each mask/score pair and apply the mask to the overlay image.
    for mask, iou_score, angle in masks_with_scores:

        # Get the bounding box of the mask
        row_indices = np.any(mask == 1, axis=1)
        col_indices = np.any(mask == 1, axis=0)
        row_min, row_max = np.where(row_indices)[0][[0, -1]]
        col_min, col_max = np.where(col_indices)[0][[0, -1]]

        # Crop the mask and resize the overlay image
        cropped_mask = mask[row_min:row_max+1, col_min:col_max+1]
        resized_overlay = cv2.resize(overlay_image, (col_max - col_min + 1, row_max - row_min + 1), interpolation=cv2.INTER_AREA)

        # Rotate the overlay image to the best angle
        M = cv2.getRotationMatrix2D((resized_overlay.shape[1]/2, resized_overlay.shape[0]/2), angle, 1)
        rotated_overlay = cv2.warpAffine(resized_overlay, M, (resized_overlay.shape[1], resized_overlay.shape[0]))


        # Apply the mask color
        iou_score = np.interp(iou_score, [0.75, 0.85, 1], [0, 0.8, 1])
        color = matplotlib.colormaps['coolwarm'](iou_score)[:3]
        color = tuple([int(c * 255) for c in color[::-1]])
        for i in range(3):
            overlay[row_min:row_max+1, col_min:col_max+1, i] = np.where(cropped_mask == 1, color[i], overlay[row_min:row_max+1, col_min:col_max+1, i])

        # Apply the resized overlay image with transparency
        if rotated_overlay.shape[2] == 4:
            alpha_overlay = rotated_overlay[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_overlay
        else:
            alpha_overlay = 1.0
            alpha_bg = 0.0

        for c in range(3):
            overlay[row_min:row_max+1, col_min:col_max+1, c] = (alpha_overlay * rotated_overlay[:, :, c] + alpha_bg * overlay[row_min:row_max+1, col_min:col_max+1, c])

    output_image = cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0)
    cv2.imwrite(output_path, output_image)

    # Display input and output images side by side
    input_image_resized = cv2.resize(input_image, (int(input_image.shape[1]/2), int(input_image.shape[0]/2)))
    output_image_resized = cv2.resize(output_image, (int(output_image.shape[1]/2), int(output_image.shape[0]/2)))
    combined_image = np.concatenate((input_image_resized, output_image_resized), axis=1)
    # cv2.imshow("Input and Output Images", combined_image)
    
    # cv2.waitKey(0)


def compare_masks(mask1, mask2, rotation_steps=36, padding=0):
    mask1_cropped = crop_mask(mask1)
    mask2_cropped = crop_mask(mask2)

    mask1_padded = cv2.copyMakeBorder(mask1_cropped, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    mask2_padded = cv2.copyMakeBorder(mask2_cropped, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    scale_factor = np.sqrt(np.sum(mask1_padded) / np.sum(mask2_padded))
    resized_mask2 = cv2.resize(mask2_padded, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    resized_mask2_cropped = crop_mask(resized_mask2)

    h1, w1 = mask1_cropped.shape[:2]
    h2, w2 = resized_mask2_cropped.shape[:2]

    # Calculate the length of the diagonal of the mask's bounding box
    diagonal = int(np.sqrt(h2**2 + w2**2))

    # If either dimension of mask1 is greater than the diagonal, resize mask1 to fit into the diagonal.
    if h1 > diagonal or w1 > diagonal:
        scale_factor_mask1 = diagonal / max(h1, w1)
        mask1_cropped = cv2.resize(mask1_cropped, None, fx=scale_factor_mask1, fy=scale_factor_mask1, interpolation=cv2.INTER_CUBIC)
        h1, w1 = mask1_cropped.shape[:2]

    # Pad the masks to create a square image with side length equal to the diagonal
    pad_h1_top = (diagonal - h1) // 2
    pad_h1_bottom = diagonal - h1 - pad_h1_top
    pad_w1_left = (diagonal - w1) // 2
    pad_w1_right = diagonal - w1 - pad_w1_left
    
    pad_h2_top = (diagonal - h2) // 2
    pad_h2_bottom = diagonal - h2 - pad_h2_top
    pad_w2_left = (diagonal - w2) // 2
    pad_w2_right = diagonal - w2 - pad_w2_left
    
    mask1_cropped = cv2.copyMakeBorder(mask1_cropped, pad_h1_top, pad_h1_bottom, pad_w1_left, pad_w1_right, cv2.BORDER_CONSTANT, value=0)
    resized_mask2_cropped = cv2.copyMakeBorder(resized_mask2_cropped, pad_h2_top, pad_h2_bottom, pad_w2_left, pad_w2_right, cv2.BORDER_CONSTANT, value=0)

    max_iou = 0
    max_angle = 0
    for angle in np.linspace(0, 360, rotation_steps):
        rotated_mask2 = np.array(Image.fromarray((resized_mask2_cropped * 255).astype(np.uint8)).rotate(-angle, resample=Image.BICUBIC))
        rotated_mask2 = (rotated_mask2 > 127).astype(np.float32)

        iou_score = iou(mask1_cropped, rotated_mask2)
        max_iou = max(max_iou, iou_score)

        if max_iou == iou_score:
            max_angle = angle

    rotated_mask2 = np.array(Image.fromarray((resized_mask2_cropped * 255).astype(np.uint8)).rotate(max_angle, resample=Image.BICUBIC))
    rotated_mask2 = (rotated_mask2 > 127).astype(np.float32)

    # cv2.imshow("mask1_cropped", mask1_cropped)
    # cv2.imshow("resized_mask2_cropped", rotated_mask2)
    # cv2.waitKey(0)

    return max_iou, max_angle


def main(input_image_path, output_masks_dir):
    # Set the paths to the reference masks directory and the output masks directory
    reference_masks_dir = config["Reference_Masks_Dir"]
    #output_masks_dir = "C:/Users/fawaz/Documents/Github/SAM/output_images/where_waldo"

    # Read the reference masks into a list
    reference_masks = []
    for ref_filename in os.listdir(reference_masks_dir):
        if ref_filename.endswith('.png'):
            ref_mask_path = os.path.join(reference_masks_dir, ref_filename)
            reference_masks.append(read_mask(ref_mask_path))

    # Create a list to store the IoU scores along with their corresponding file paths
    iou_scores = []
    # Iterate through the output masks and compute IoU scores
    for filename in os.listdir(output_masks_dir):
        if filename.endswith('.png'):
            mask_path = os.path.join(output_masks_dir, filename)
            mask = read_mask(mask_path)
            

            mask_iou_scores = []
            mask_angle_scores = []
            for reference_mask in reference_masks:
                iou_score, mask_angle = compare_masks(reference_mask, mask)
                mask_iou_scores.append(iou_score)
                mask_angle_scores.append(mask_angle)

            max_iou_score = max(mask_iou_scores)
            best_angle = mask_angle_scores[mask_iou_scores.index(max_iou_score)]
            avg_iou_score = sum(mask_iou_scores) / len(mask_iou_scores)
            iou_scores.append((max_iou_score, avg_iou_score, best_angle, mask_path))
            print(f"Comparing {mask_path} with reference masks... iou_score: {max_iou_score}, avg_iou_score: {avg_iou_score}")

    # Sort the list in descending order based on the max IoU scores
    # sorted_iou_scores = sorted(iou_scores, key=lambda x: x[0], reverse=True)
    input_image = cv2.imread(input_image_path)

    masks_with_scores = []
    for max_iou_score, avg_iou_score, best_angle, mask_path in iou_scores:
        if max_iou_score > 0.77 and avg_iou_score > 0.72:
            mask = read_mask(mask_path)
            masks_with_scores.append((mask, max_iou_score, best_angle))

    overlay_image_path = config["Cover_Image_Dir"]
    output_image_path = os.path.join(config["Output_Path"], "combined_output_" + str((os.path.basename(input_image_path))[:-4]) + ".png")
    generate_output_image(input_image, masks_with_scores, overlay_image_path, output_image_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mask_comparison.py <input_image> <output_masks_dir>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_masks_dir = sys.argv[2]
    main(input_image_path, output_masks_dir)