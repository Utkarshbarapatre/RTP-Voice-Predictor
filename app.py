from flask import Flask, render_template, request, redirect, jsonify, session, url_for, flash, send_file
import numpy as np
import pickle
import soundfile
import librosa
import os
from werkzeug.utils import secure_filename
from fpdf import FPDF
import time
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from flask_session import Session
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import re

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "your_secret_key"
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'barapatreutkarsh2003@gmail.com' 
app.config['MAIL_PASSWORD'] = 'wyxklgkxyoxcvqrm'  
app.config['MAIL_DEFAULT_SENDER'] = 'barapatreutkarsh2003@gmail.com'
mail = Mail(app)
Session(app)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@app.route('/send_email_report/<email>')
def send_email_report(email):
    if not email:
        return jsonify({"success": False, "message": "Error: No email provided"}), 400
    
    # ✅ Retrieve prediction data from session
    username = session.get("username", "Unknown")
    prediction = session.get("prediction", "Unknown")
    stress_level = session.get("stress_level", "Unknown")
    time_taken = session.get("time_taken", 0.0)

    # ✅ Define PDF file path
    pdf_folder = "static/reports"
    os.makedirs(pdf_folder, exist_ok=True)  # Ensure folder exists
    pdf_path = os.path.join(pdf_folder, f"stress_report_{username}.pdf")

    # ✅ Generate PDF report
    c = canvas.Canvas(pdf_path, pagesize=(612, 792))  # Letter size (8.5x11 inches)
    c.setFont("Helvetica-Bold", 16)  
    c.drawString(200, 750, "Stress Predict Report")  

    c.setFont("Helvetica", 12)
    y_position = 720  
    c.drawString(100, y_position, f"UserID: {username}")

    y_position -= 20
    c.drawString(100, y_position, f"Email: {email}") 

    y_position -= 40  # Extra space before prediction
    c.drawString(100, y_position, f"Prediction: {prediction}")


    y_position -= 20
    c.drawString(100, y_position, f"Stress Level: {stress_level}")

    y_position -= 20
    c.drawString(100, y_position, f"Time Taken: {time_taken} seconds")

    y_position -= 20
    c.drawString(100, y_position, f"Consultant Name: Dr. Vidyadhar Bapat")

    y_position -= 20
    c.drawString(100, y_position, f"Consultant Address: D 2, Dhanaraj Apartments, Apte Road,Pune 411004")
   
    y_position -= 20
    c.drawString(100, y_position,f"Phone Number: +91 9850415170")


    c.save()

    # ✅ Create and send the email
    subject = "Your Stress Prediction Report"
    msg = Message(subject, recipients=[email])
    msg.body = "Please find your stress prediction report attached."

    # ✅ Attach the generated PDF
    with open(pdf_path, "rb") as pdf_file:
        msg.attach(f"Stress_Report_{username}.pdf", "application/pdf", pdf_file.read())

    try:
        mail.send(msg)
        return jsonify({"success": True, "message": "Email with report sent successfully!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error sending email: {str(e)}"})



# Function to remove emojis from text to avoid PDF encoding issues
def remove_emojis(text):
    if not text:
        return text
    # Pattern to match emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+"
    )
    return emoji_pattern.sub(r'', text)


# Updated function to send team performance email with better error handling
@app.route('/send_team_performance_email/<email>')
def send_team_performance_email(email):
    if not email:
        return jsonify({"success": False, "message": "Error: No email provided"}), 400
    
    try:
        # Check if team data exists in session
        if 'team_data' not in session:
            return jsonify({"success": False, "message": "No team performance data available"}), 400
        
        # Get data from session
        data = session.get('team_data', {})
        parameter_ratings = session.get('parameter_ratings', {})
        
        # Generate PDF report for email
        try:
            pdf_path = generate_team_performance_pdf(email, data, parameter_ratings)
        except Exception as pdf_error:
            print(f"PDF Generation Error: {str(pdf_error)}")
            return jsonify({"success": False, "message": f"Error generating PDF: {str(pdf_error)}"}), 500
        
        # Create and send the email
        subject = "Your Team Performance Report"
        
        try:
            msg = Message(subject, recipients=[email])
            msg.body = """
            Hello,
            
            Please find your team performance report attached. This report includes all the metrics and evaluations from your recent team performance prediction.
            
            Thank you for using our service!
            """
            
            # Attach the generated PDF
            with open(pdf_path, "rb") as pdf_file:
                msg.attach("Team_Performance_Report.pdf", "application/pdf", pdf_file.read())
            
            mail.send(msg)
            
            # Store email in session for future use
            session['user_email'] = email
            return jsonify({"success": True, "message": "Email with team performance report sent successfully!"})
        
        except Exception as email_error:
            print(f"Email Sending Error: {str(email_error)}")
            return jsonify({"success": False, "message": f"Error sending email: {str(email_error)}"}), 500
    
    except Exception as general_error:
        print(f"General Error: {str(general_error)}")
        return jsonify({"success": False, "message": f"An unexpected error occurred: {str(general_error)}"}), 500

# Updated function to generate team performance PDF without using Helvetica-Italic
def generate_team_performance_pdf(email, data, parameter_ratings):
    pdf_folder = "static/reports"
    os.makedirs(pdf_folder, exist_ok=True)
    pdf_path = os.path.join(pdf_folder, f"team_report_for_email_{email.replace('@', '_at_')}.pdf")
    
    # Create PDF using ReportLab
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 50, "Team Performance Report")
    
    # Team details
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 100, "Team Details")
    
    c.setFont("Helvetica", 12)
    y_position = height - 130
    
    # Basic team information
    team_name = session.get('team_name', 'Not specified')
    c.drawString(50, y_position, f"Team Name: {team_name}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Number of Members: {data.get('Team_Members', 'N/A')}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Experience Level: {data.get('Experience_Level', 'N/A')}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Skill Diversity: {data.get('Skill_Diversity', 'N/A')}")
    y_position -= 20
    
    # Performance metrics section
    y_position -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Performance Metrics")
    y_position -= 30
    
    c.setFont("Helvetica", 12)
    
    # Core metrics
    c.drawString(50, y_position, f"Productivity Score: {data.get('Productivity_Score', 'N/A')}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Quality Score: {data.get('Quality_Score', 'N/A')}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Collaboration Score: {data.get('Collaboration_Score', 'N/A')}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Efficiency Ratio: {data.get('Efficiency_Ratio', 'N/A')}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Task Completion Rate: {data.get('Task_Completion_Rate', 'N/A')}%")
    y_position -= 20
    
    c.drawString(50, y_position, f"Error Rate: {data.get('Error_Rate', 'N/A')}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Conflict Occurrences: {data.get('Conflict_Occurrences', 'N/A')}")
    y_position -= 20
    
    c.drawString(50, y_position, f"Engagement Level: {data.get('Engagement_Level', 'N/A')}")
    y_position -= 20
    
    # Optional parameters section
    y_position -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Additional Parameters")
    y_position -= 30
    
    c.setFont("Helvetica", 12)
    
    # Check if optional parameters exist and add them
    team_satisfaction = data.get('Team_Satisfaction', 'Not provided')
    c.drawString(50, y_position, f"Team Satisfaction Index: {team_satisfaction}")
    y_position -= 20
    
    communication_score = data.get('Communication_Score', 'Not provided')
    c.drawString(50, y_position, f"Team Communication Score: {communication_score}")
    y_position -= 20
    
    deadline_adherence = data.get('Deadline_Adherence', 'Not provided')
    c.drawString(50, y_position, f"Deadline Adherence: {deadline_adherence}%")
    y_position -= 20
    
    manager_evaluation = data.get('Manager_Evaluation', 'Not provided')
    c.drawString(50, y_position, f"Manager's Evaluation Score: {manager_evaluation}")
    y_position -= 20
    
    # Parameter ratings section
    y_position -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Parameter Ratings")
    y_position -= 30
    
    c.setFont("Helvetica", 12)
    
    if parameter_ratings:
        for param, rating in parameter_ratings.items():
            c.drawString(50, y_position, f"{param}: {rating}")
            y_position -= 20
    
    # Overall result
    y_position -= 20
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "Prediction Result")
    y_position -= 30
    
    # Remove emojis from prediction result to avoid encoding issues
    prediction_result = remove_emojis(data.get('prediction', 'N/A'))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, f"Team Performance: {prediction_result}")
    
    # Calculate overall performance score
    y_position -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Overall Performance Score")
    y_position -= 20
    
    # Calculate performance score based on all metrics including optional ones
    performance_score = (
        float(data.get('Productivity_Score', 5)) + 
        float(data.get('Quality_Score', 5)) + 
        float(data.get('Collaboration_Score', 5)) + 
        float(data.get('Efficiency_Ratio', 5)) + 
        float(data.get('Team_Satisfaction', 5)) +
        float(data.get('Communication_Score', 5)) +
        float(data.get('Manager_Evaluation', 5))
    ) / 7  # Average of 7 metrics
    
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, f"Score (out of 10): {performance_score:.2f}")
    

    y_position = 50  # Bottom of page
    c.setFont("Helvetica", 10)  # Changed from Helvetica-Italic to regular Helvetica
    c.drawString(50, y_position, f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, y_position - 15, f"For: {email}")
    
    c.save()
    return pdf_path

@app.route('/download_team_performance_report')
def download_team_performance_report():
    if 'team_data' not in session:
        return "No team performance data available. Please make a prediction first.", 400
    
    data = session.get('team_data', {})
    parameter_ratings = session.get('parameter_ratings', {})
    email = session.get('user_email', 'user@example.com')
    
    pdf_path = generate_team_performance_pdf(email, data, parameter_ratings)
    
    return send_file(pdf_path, as_attachment=True, download_name="team_performance_report.pdf")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            flash('Login successful!', 'success')

            session['user_id'] = user.id
            session['user_email'] = user.email  

            return render_template('home.html')
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html')


@app.route('/about', methods=["GET","POST"])
def about():
    return render_template('about.html')

@app.route('/', methods=["GET","POST"])
def myhome():
    return render_template('myhome.html')


@app.route('/contact', methods=["GET","POST"])
def contact():
    return render_template('contact.html')


@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match.', 'danger')

    return render_template('register.html')

@app.route('/logout/')
def logout():
    session.clear() 
    return redirect(url_for('login'))


app.config["SECRET_KEY"] = "speechemotionkey"

observed_emotions = ['calm', 'happy', 'fearful', 'disgust', 'neutral','angry','sad']

pre =  ""
em = ""

emotion_emoji = {
    "calm": "😌",
    'happy': "😃", 
    'fearful': "😨", 
    'disgust':"🤢",
    "angry" : "😡",
    "neutral" : "😐",
    'sad':"😞"
}

stress_mapping = {
    "calm": "Low",
    "happy": "Low",
    "neutral": "Medium",
    "sad": "High",
    "angry": "Very High",
    "fearful": "High",
    "disgust": "Medium"
}

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


model_name = "modelhybridEmotion.pkl" 
ml_model = pickle.load(open(model_name,"rb"))


@app.route('/home', methods=["GET","POST"])
def home():
    return render_template('home.html')


def generate_charts(data):
    charts = {}
    
    plt.figure(figsize=(8, 5))
    metrics = ['Productivity', 'Quality', 'Collaboration', 'Efficiency']
    values = [data['Productivity_Score'], data['Quality_Score'], 
              data['Collaboration_Score'], data['Efficiency_Ratio']]
    
    plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FFC107', '#9C27B0'])
    plt.ylim(0, 10)
    plt.title('Key Performance Metrics')
    plt.tight_layout()
   
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    charts['bar_chart'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    categories = ['Experience', 'Skill Diversity', 'Training', 'Engagement', 'Communication']
    
    values = [
        data['Experience_Level'],
        data['Skill_Diversity'],
        data['Training_Hours']/10,  # Normalize to 0-10 scale
        data['Engagement_Level']/10,  # Normalize to 0-10 scale
        data.get('Communication_Score', 5)  # Default to 5 if not provided
    ]
    
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1] 
    angles += angles[:1]  
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='#2196F3')
    ax.fill(angles, values, alpha=0.25, color='#2196F3')
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 10)
    ax.grid(True)
    
    # Save to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    charts['radar_chart'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Create a gauge chart for overall performance
    # Calculate performance score based on all metrics including optional ones
    performance_score = (
        data['Productivity_Score'] + 
        data['Quality_Score'] + 
        data['Collaboration_Score'] + 
        data['Efficiency_Ratio'] + 
        data.get('Team_Satisfaction', 5) +  # Default to 5 if not provided
        data.get('Communication_Score', 5) +  # Default to 5 if not provided
        data.get('Manager_Evaluation', 5)  # Default to 5 if not provided
    ) / 7  # Average of 7 metrics
    
    # Create gauge chart
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Draw the gauge
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    ax.axvspan(0, 3.33, color='#FF5252', alpha=0.3)
    ax.axvspan(3.33, 6.66, color='#FFC107', alpha=0.3)
    ax.axvspan(6.66, 10, color='#4CAF50', alpha=0.3)
    
   
    ax.arrow(0, 0, performance_score, 0, head_width=0.1, head_length=0.3, fc='black', ec='black')
    
    
    ax.text(1.5, 0.25, 'Poor', ha='center', fontsize=12)
    ax.text(5, 0.25, 'Average', ha='center', fontsize=12)
    ax.text(8.5, 0.25, 'Excellent', ha='center', fontsize=12)
    ax.text(performance_score, 0.5, f'{performance_score:.1f}', ha='center', fontsize=14, fontweight='bold')
    
    plt.title('Overall Performance Score')
    plt.tight_layout()
    

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    charts['gauge_chart'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    labels = ['Productivity', 'Quality', 'Collaboration', 'Efficiency']
    sizes = [data['Productivity_Score'], data['Quality_Score'], 
             data['Collaboration_Score'], data['Efficiency_Ratio']]
    
   
    if 'Deadline_Adherence' in data and data['Deadline_Adherence'] is not None:
        labels.append('Deadline Adherence')
        sizes.append(data['Deadline_Adherence']/10)  # Normalize to similar scale
    

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#FF5722'])
    plt.axis('equal')
    plt.title('Contribution Factors')
   
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    charts['pie_chart'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return charts


# Function to evaluate parameters and provide ratings
def evaluate_parameters(data):
    ratings = {}
    
    # Evaluate Skill Diversity
    skill_diversity = float(data['Skill_Diversity'])
    if skill_diversity >= 7:
        ratings['Skill Diversity'] = 'High'
    elif skill_diversity >= 4:
        ratings['Skill Diversity'] = 'Moderate'
    else:
        ratings['Skill Diversity'] = 'Low'
    
    # Evaluate Productivity Score
    productivity = float(data['Productivity_Score'])
    if productivity >= 7:
        ratings['Productivity Score'] = 'High'
    elif productivity >= 4:
        ratings['Productivity Score'] = 'Moderate'
    else:
        ratings['Productivity Score'] = 'Low'
    
    # Evaluate Conflict Occurrences
    conflicts = int(data['Conflict_Occurrences'])
    if conflicts <= 1:
        ratings['Conflict Occurrences'] = 'Low'
    elif conflicts <= 3:
        ratings['Conflict Occurrences'] = 'Moderate'
    else:
        ratings['Conflict Occurrences'] = 'High'
    
    # Overall rating based on prediction
    if data['prediction'] == 'Good':
        ratings['Overall'] = 'Good'
    else:
        ratings['Overall'] = 'Needs Improvement'
    
    return ratings


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            model = pickle.load(open('model.pkl', 'rb'))

            # Store team name if provided
            team_name = request.form.get('team_name', '')
            if team_name:
                session['team_name'] = team_name

            # Extract form data from the request
            new_data = {
                "Number of Members": [int(request.form.get('Team_Members', 0))],
                "Role Distribution": [int(request.form.get('Role_Distribution', 0))],
                "Experience Level": [float(request.form.get('Experience_Level', 0))],
                "Skill Diversity": [float(request.form.get('Skill_Diversity', 0))],
                "Task Completion Rate": [float(request.form.get('Task_Completion_Rate', 0))],
                "Error Rate": [float(request.form.get('Error_Rate', 0))],
                "Productivity Score": [float(request.form.get('Productivity_Score', 0))],
                "Quality Score": [float(request.form.get('Quality_Score', 0))],
                "Efficiency Ratio": [float(request.form.get('Efficiency_Ratio', 0))],
                "Collaboration Score": [float(request.form.get('Collaboration_Score', 0))],
                "Conflict Occurrences": [int(request.form.get('Conflict_Occurrences', 0))],
                "Engagement Level": [float(request.form.get('Engagement_Level', 0))],
                "Training Hours": [float(request.form.get('Training_Hours', 0))],
                "Revenue Generated": [float(request.form.get('Revenue_Generated', 0))],
                "Customer Satisfaction": [float(request.form.get('Customer_Satisfaction', 0))],
                "Milestones Achieved": [int(request.form.get('Milestones_Achieved', 0))],
                "Average Time to Complete Task": [float(request.form.get('Time_to_Complete', 0))],
                "Overtime Hours": [float(request.form.get('Overtime_Hours', 0))],
                "Planned vs. Actual Hours": [float(request.form.get('Planned_vs_Actual', 0))],
            }
            
            # Get optional parameters
            team_satisfaction = request.form.get('Team_Satisfaction')
            communication_score = request.form.get('Communication_Score')
            deadline_adherence = request.form.get('Deadline_Adherence')
            manager_evaluation = request.form.get('Manager_Evaluation')
            
            # Convert the input data into a DataFrame
            input_df = pd.DataFrame(new_data)
            
            # Make predictions
            predictions = model.predict(input_df)
            
            # Determine result
            result = 'Not Much Good but can be improved 😞' if predictions[0] == 0 else 'Good'
            
            # Create a data dictionary for charts and parameter ratings
            data = {
                'Team_Members': float(request.form.get('Team_Members', 0)),
                'Role_Distribution': float(request.form.get('Role_Distribution', 0)),
                'Experience_Level': float(request.form.get('Experience_Level', 0)),
                'Skill_Diversity': float(request.form.get('Skill_Diversity', 0)),
                'Task_Completion_Rate': float(request.form.get('Task_Completion_Rate', 0)),
                'Error_Rate': float(request.form.get('Error_Rate', 0)),
                'Productivity_Score': float(request.form.get('Productivity_Score', 0)),
                'Quality_Score': float(request.form.get('Quality_Score',   0)),
                'Quality_Score': float(request.form.get('Quality_Score', 0)),
                'Efficiency_Ratio': float(request.form.get('Efficiency_Ratio', 0)),
                'Collaboration_Score': float(request.form.get('Collaboration_Score', 0)),
                'Conflict_Occurrences': int(request.form.get('Conflict_Occurrences', 0)),
                'Engagement_Level': float(request.form.get('Engagement_Level', 0)),
                'Training_Hours': float(request.form.get('Training_Hours', 0)),
                'Revenue_Generated': float(request.form.get('Revenue_Generated', 0)),
                'Customer_Satisfaction': float(request.form.get('Customer_Satisfaction', 0)),
                'Milestones_Achieved': int(request.form.get('Milestones_Achieved', 0)),
                'Time_to_Complete': float(request.form.get('Time_to_Complete', 0)),
                'Overtime_Hours': float(request.form.get('Overtime_Hours', 0)),
                'Planned_vs_Actual': float(request.form.get('Planned_vs_Actual', 0)),
                'prediction': result
            }
            
            # Add optional parameters if provided
            if team_satisfaction:
                data['Team_Satisfaction'] = float(team_satisfaction)
            if communication_score:
                data['Communication_Score'] = float(communication_score)
            if deadline_adherence:
                data['Deadline_Adherence'] = float(deadline_adherence)
            if manager_evaluation:
                data['Manager_Evaluation'] = float(manager_evaluation)
            
            # Generate parameter ratings
            parameter_ratings = evaluate_parameters(data)
            
            # Generate charts
            charts = generate_charts(data)
            
            # Store in session for report generation
            session['team_data'] = data
            session['parameter_ratings'] = parameter_ratings
            
            return render_template('predict.html', 
                                  result=result, 
                                  parameter_ratings=parameter_ratings,
                                  charts=charts)

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('predict.html')


# Original download report function (kept as is)
@app.route('/generate_report')
def download_report():
    if "prediction" not in session:
        return "No prediction available. Please make a prediction first.", 400

    pdf_path = generate_report(
        session.get("username", "Unknown"),  # ✅ Fetch from session
        session.get("email", "Unknown"),     # ✅ Fetch from session
        session["prediction"], 
        session["emoji"], 
        session["stress_level"], 
        session["time_taken"]
    )
    return send_file(pdf_path, as_attachment=True)




@app.route('/prediction', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start_time = time.time()

        if "audio-file" not in request.files:
            flash("No file uploaded!", "error")
            return redirect(request.url)

        file = request.files["audio-file"]
        if file.filename == "":
            flash("Invalid file name!", "error")
            return redirect(request.url)

        if file:
            features = extract_feature(file_name=file, mfcc=True, chroma=True, mel=True)
            features = features.reshape(1, -1)
            prediction = ml_model.predict(features)[0]

            end_time = time.time()
            time_taken = round(end_time - start_time, 2)

            stress_level = stress_mapping.get(prediction, "Unknown")
            emoji = emotion_emoji.get(prediction, "")

            # ✅ Fetch from session instead of form
            username = session.get("user_id", "Unknown")
            email = session.get("user_email", "Unknown")

            session["username"] = username
            session["email"] = email
            session["prediction"] = prediction.capitalize()
            session["emoji"] = emoji
            session["stress_level"] = stress_level
            session["time_taken"] = time_taken

            print("Session Data:", session)

    return render_template(
        'prediction.html',
        username=session.get("username", "Unknown"),
        email=session.get("email", "Unknown"),
        prediction=session.get("prediction", ""),
        emoji=session.get("emoji", ""),
        stress_level=session.get("stress_level", ""),
        time_taken=session.get("time_taken", 0.0)
    )


@app.route('/realtimeprediction', methods=["GET","POST"])
def audio():
    prediction1 = ""
    emoji1 = ""
    stress_level = ""
    time_taken = 0.0

    if request.method == "POST":
        start_time = time.time()

        print("Form Data received")
        if "file" not in request.files:
            print("1")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            print("2")
            return redirect(request.url)

        if file:
            features = extract_feature(file_name=file, mfcc=True, chroma=True, mel=True)
            features = features.reshape(1, -1)
            prediction1 = ml_model.predict(features)[0]

            end_time = time.time()
            time_taken = round(end_time - start_time, 2)

            stress_level = stress_mapping.get(prediction1, "Unknown")
            emoji1 = emotion_emoji.get(prediction1, "")

            print(prediction1, emoji1, stress_level, time_taken)

    global pre, em, global_stress, global_time
    pre = prediction1
    em = emoji1
    global_stress = stress_level
    global_time = time_taken
    

    return render_template(
        "realtimepredection.html",
        prediction1=prediction1,
        emoji1=emoji1,
        stress_level=stress_level,
        time_taken=time_taken
    )

    

@app.route('/redirect', methods=["GET", "POST"])
def red():
    predict = pre
    emoj = em
    stress = global_stress  # Add stress level
    time_taken = global_time  # Add time taken

    return render_template(
        "redirectprediction.html",
        pred=predict,
        emo=emoj,
        stress_level=stress,
        time_taken=time_taken
    )


# Original generate report function (kept as is)
@app.route('/generate_report')
def generate_report(username, email, prediction, emoji, stress_level, time_taken):
    pdf_path = "static/reports/prediction_report.pdf"
    os.makedirs("static/reports", exist_ok=True)  # Ensure folder exists

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)  
    c.drawString(180, 770, "Stress Predict Report")  

    c.setFont("Helvetica", 12)
    c.drawString(100, 740, f"UserID: {username}")
    c.drawString(100, 720, f"Email: {email}") 
    c.drawString(100, 690, f"Prediction: {prediction}")
    c.drawString(100, 650, f"Stress Level: {stress_level}")
    c.drawString(100, 630, f"Time Taken: {time_taken} seconds")
    c.drawString(100, 610, f"Consultant Name: Dr. Vidyadhar Bapat")
    c.drawString(100, 590, f"Consultant Address: D 2, Dhanaraj Apartments, Apte Road,Pune 411004")
    c.drawString(100, 570, f"Phone Number: +91 9850415170")
    

    c.save()
    return pdf_path
    

if __name__ == '__main__':
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/reports", exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)