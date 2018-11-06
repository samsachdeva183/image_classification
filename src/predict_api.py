import requests
import base64


with open("cricket-test.jpg", "rb") as image_file:
    encoded_string = image_file.read()

url = "http://" + "x.x.x.x" + ":5000/imageclass"


files = {'file' : open("cricket-test.jpg", "rb")}
response = requests.post(url, files = files)
print (response.content)

