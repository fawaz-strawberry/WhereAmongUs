import sys
import subprocess
import yaml

config = yaml.safe_load(open("C:/Users/fawaz/Documents/Github/WhereAmongUs/configuration.yaml"))

def run_segmentation(input_image):

    # Replace this line with the actual command to run your segmentation script
    command = f"python " + config["Mask_Generator_Path"] + " --checkpoint " + config["Checkpoint_Path"] + f" --input {input_image} --output " + config["Output_Path"]
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_segmentation.py <input_image>")
        sys.exit(1)

    input_image = sys.argv[1]
    print(input_image)
    run_segmentation(input_image)

    
