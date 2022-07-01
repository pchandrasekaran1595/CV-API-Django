### **Computer Vision API using Django**

<br>

- Install python 3.9.x
- Run `pip install virtualenv`
- Run `make-env.bat`
- Run `start-local-server.bat`. The API will now be served at `http://127.0.0.1:10000`
- To run in production mode
    - Change `DEBUG` to `False` in Main.settings
    - Ensure appropriate environment variable is set
    - Run `collect-static.bat` before `start-local-server.bat`
- Heroku URL : `https://pcs-cv-api.herokuapp.com/`

<br>

**Endpoints**

1. `classify/` - returns highest confidence prediction label
2. `detect/` &nbsp;&nbsp;&nbsp; - returns highest confidence bounding box and associated label
3. `segment/` &nbsp; - returns list of labels and base64 encoded image data

<br>
