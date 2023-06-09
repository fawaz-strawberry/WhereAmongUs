from flask import Flask, request, send_file
import os
import subprocess

app = Flask(__name__)

activation_server = "C:/Users/fawaz/Documents/Github/SAM/Scripts/activate.bat"
runner_script = "C:/Users/fawaz/Documents/Github/WhereAmongUs/runner.bat"
uploads_folder = "C:/Users/fawaz/Documents/Github/WhereAmongUs/Input_Images/"
output_folder = "C:/Users/fawaz/Documents/Github/WhereAmongUs/Output_Images/"


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file:
        filename = file.filename
        print("Saving file")
        print(filename)
        filepath = os.path.join(uploads_folder, filename)
        print("Current Filepath")
        print(filepath)
        file.save(filepath)

        # activate the environment and run the batch script
        subprocess.run([activation_server, "&", runner_script, filepath], shell=True)
        filename = os.path.basename(filename)
        output_image_path = output_folder + "combined_output_" + (filename)[:-4] + ".png"
        print("Output Image: " + output_image_path)

        return send_file(output_image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)
