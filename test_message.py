import requests

url = "http://0.0.0.0:5000/process_image"
output_path = "C:/Users/fawaz/Documents/Github/WhereAmongUs/API_Returns/" 
file_path = "C:/Users/fawaz/Documents/Github/WhereAmongUs/Input_Images/basketball-game-5.png"

# Open file in binary mode
with open(file_path, "rb") as image_file:
    # Prepare data
    file_data = {
        "file": (file_path, image_file, "image/png"),
    }
    # Send POST request with file
    response = requests.post(url, files=file_data)

# Save the response content as an image
if response.status_code == 200:
    with open(output_path + "output.png", "wb") as output_image:
        output_image.write(response.content)
        
