from time import sleep
import requests
import base64
import cv2

# URL = "http://127.0.0.1:8002"
URL = "https://mask.ahmed-amer.tech"
cam = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cam.read()

        _, im_arr = cv2.imencode('.jpg', frame)
        im_bytes = im_arr.tobytes()
        my_string = base64.b64encode(im_bytes)
            
        payload = {
            "image": my_string.decode('ascii')
        }

        res = requests.post(URL, json=payload, verify=False)
        if res.status_code == 200:
            print(res.json()) 
        print(res.status_code)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)
    
    sleep(5)
    
cam.release()
cv2.destroyAllWindows()