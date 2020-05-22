from flask import Flask,render_template, request, flash, request, redirect, url_for, send_from_directory
import os
from utils import *
from PIL import Image
from flask_scss import Scss
from werkzeug.utils import secure_filename
import utils

app = Flask(__name__)
UPLOAD_FOLDER = 'img/2-in-1_Space_Dye_Athletic_Tank/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# INPUT_FILE_NAME = "input.jpg"
OUTPUT_FILE_NAME = "output.jpg"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser alsos
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("input_file_path: ", input_file_path)
            file.save(input_file_path)
            '''
                Input: image
                output: out_images folder
            '''
            my_path = utils.recommend(input_file_path)
            return render_template("success.html")

    return render_template("home.html", data=[{'categ':'categ1'}, {'categ':'categ2'}])


# about subsite that show basic information about project and team member.
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run()
