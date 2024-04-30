from flask import Flask, render_template,request
from threading import Thread
from tk import open

# Create the Flask app
app = Flask(__name__)

# Define a route for the home page

# Configure static folder
app.static_folder = 'static'
@app.route('/')
def home():
    return render_template('index.html')

def tkkinter(user):
    from login import login
    login()
    # root.attributes
    # root.mainloop()
    

@app.route('/open_tkinter')
def open_tkinter():
    user = "hem"
    thread = Thread(target=tkkinter, args=(user,))
    thread.start()
    return render_template('index.html')

@app.route('/details',)
def details():
    detail_value = request.args.get('detail')
    if detail_value:
        # json_file_path = f"static/{detail_value}/data.json"
        raw_image = f"static/{detail_value}/raw_image.jpg"
        preprocessed_image =f"static/{detail_value}/grayscale_image.jpg" 
        score =  request.args.get('s')
        class_name = request.args.get('cl')

        return render_template("result.html",raw = raw_image, prep = preprocessed_image,score=score,class_name=class_name)
    else:
        return "Detail value not provided in the URL"
 



# Run the app if this script is executed
if __name__ == '__main__':
    app.run(debug=True)
