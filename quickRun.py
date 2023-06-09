import sys
import subprocess
import os

def run_segmentation(input_image):

    # Replace this line with the actual command to run your segmentation script
    command = f"python C:/Users/fawaz/Documents/Github/SAM/segment-anything/scripts/amg.py --checkpoint C:/Users/fawaz/Documents/Github/SAM/models/sam_vit_h_4b8939.pth --input {input_image} --output Output_Images/"
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_segmentation.py <input_image>")
        sys.exit(1)

    input_image = sys.argv[1]
    run_segmentation(input_image)

    
