'''
The following Code's goal is to performa a massive download of numerous images
based upon a given search term within a list. The images will be downloaded from
Google Images and will be stored in a folder named after the search term.
'''

# Importing the necessary libraries
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import os
import urllib.request
import base64
from PIL import Image


# Defining the path to the chromedriver.exe file
gecko_path = r"C:/Users/fawaz/Documents/GeckoDriver/geckodriver.exe"
search_terms = ["Dense Forest Area"]
images_to_download = 100

# Defining the path to the folder where the images will be stored
Main_Folder = r"C:/Users/fawaz/Documents/Github/WhereAmongUs/Downloaded_Images"

# Defining the path to the chromedriver.exe file
driver = webdriver.Firefox()

for search_term in search_terms:
    path_to_folder = os.path.join(Main_Folder, ''.join(e for e in (search_term) if e.isalnum()))

    # Creating the folder where the images will be stored
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

    # Opening the Google Images website
    driver.get("https://images.google.com/")
    element = driver.find_element(By.NAME, "q")
    element.send_keys(search_term + Keys.ENTER)

    ImageSet = []
    time.sleep(5)

    image_elements = driver.find_elements(By.TAG_NAME, "img")
    print(len(image_elements))
    image_elements = image_elements[:images_to_download]

    image_num = 0
    for image_element in image_elements:
        image_path = path_to_folder + "/" + ''.join(e for e in (search_term + str(image_num)) if e.isalnum()) + ".jpg"
        g = open(image_path, "wb")
        element_src = image_element.get_attribute("src")
        if(element_src == None):
            continue
        if("https://" in element_src or "http://" in element_src):
            continue
        split_element_src = element_src.split(",")
        if len(split_element_src) > 1:
            element_src = split_element_src[1]
        else:
            continue
        g.write(base64.b64decode(element_src))
        print(image_element.get_attribute("src")[:10])
        g.close()

        # Reopen and Upscale the image
        image = Image.open(image_path).convert("RGB")

        # Get the image size
        width, height = image.size

        # Resize the image
        if(width < 512):
            image = image.resize((512, int(height * (512 / width))), Image.ANTIALIAS)
        if(height < 512):
            image = image.resize((int(width * (512 / height)), 512), Image.ANTIALIAS)

        image.save(image_path)

        print(image_path)

        os.system("C:/Users/fawaz/Documents/Github/WhereAmongUs/Scripts/OverlayGenerator/runner.bat " + image_path)
        image_num += 1



time.sleep(5)
driver.close()
