import cv2
import requests
import numpy as np

class OpenFiles():
    def __init__(self):
        self.files = []
    def open(self, file_name, mode):
        f = open(file_name,mode)
        self.files.append(f)
        return f
    def close(self):
        list(map(lambda f: f.close(), self.files))

def get_data(url, token, last_id):
    """Get data from app"""
    params = {
        'token': token,
        'last_id': last_id
    }
    try:
        response = requests.get(url,params=params).json()
    except requests.exceptions.ConnectionError:
        return None
    return response

def send_data(url,token,img_id,files):
    """Send data to app"""
    data = {
        'id': img_id
    }
    headers = {
        'token': token
    }
    try:
        response = requests.post(url,data=data,files=files,headers=headers).json()
    except requests.exceptions.ConnectionError:
        return False
    return response

def url2img(url):
    """Convert url to image"""
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()),dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image