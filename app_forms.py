from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.debug = True

# Adding configuration for using a SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

# Creating an SQLAlchemy instance
db = SQLAlchemy(app)

# Models
class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(20), nullable=False)
    last_name = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"Name: {self.first_name}, Age: {self.age}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    age = request.form['age']

    profile = Profile(first_name=first_name, last_name=last_name, age=age)
    db.session.add(profile)
    db.session.commit()

    return redirect('/')

if __name__ == '__main__':
    # Creating the database tables (if they don't exist)
    db.create_all()
    app.run()
