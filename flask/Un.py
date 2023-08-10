from flask import Flask,render_template
import pandas as pd
import os
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {"csv"}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
name = []
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            name.append(filename)
            #return redirect(url_for('download_file', name=filename))
            return redirect(url_for("csvtohtml",name=filename))
    return render_template('accordion.html')
    #'''
    #<!doctype html>
    #<title>Upload new File</title>
    #<h1>Upload new File</h1>
    #<form method=post enctype=multipart/form-data>
    #  <input type=file name=file>
    #  <input type=submit value=Upload>
    #</form>
    #'''
from flask import send_from_directory

@app.route('/uploads/<name>')
#def download_file(name):
 #   return send_from_directory(app.config["UPLOAD_FOLDER"], name)
    
#@app.route('/uploads/<name>')
def csvtohtml(name):
    data = pd.read_csv(name)
    data = pd.DataFrame(data)
    return render_template('index.html', tables=[data.to_html(header="true" , 
                                               table_id="table")],titles=["data.index"] )    
@app.route('/home')
def home():
    return render_template('accordion.html')
if __name__ == "__main__":
    app.run(host="localhost",port=int(5000))
 