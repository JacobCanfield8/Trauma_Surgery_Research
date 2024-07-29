import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import joblib
import numpy as np
import qrcode

app = Flask(__name__)

# Load the trained model
model = joblib.load('xgb_model.pkl')

# Define the order of features
feature_order = ['SEX', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE']

# Define mappings
sex_mapping = {'Male': 1, 'Female': 2}
prehospital_mapping = {'yes': 1, 'no': 2}
trauma_mapping = {
    'Blunt': 1,
    'Penetrating': 2,
    'Burn': 3,
    'Other/unspecified': 4
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form and map categorical inputs
        input_data = [
            sex_mapping[request.form['SEX']],
            int(request.form['EMSSBP']),
            int(request.form['EMSPULSERATE']),
            int(request.form['EMSRESPIRATORYRATE']),
            int(request.form['EMSTOTALGCS']),
            prehospital_mapping[request.form['PREHOSPITALCARDIACARREST']],
            trauma_mapping[request.form['TRAUMATYPE']]
        ]
        # Convert the input data to a numpy array
        input_data = np.array(input_data).reshape(1, -1)
        # Make prediction and get probability
        prediction = int(model.predict(input_data)[0])
        probability = model.predict_proba(input_data)[0]
        confidence = np.max(probability) * 100  # Convert to percentage
        # Convert prediction to a descriptive result
        result = "Deceased" if prediction == 1 else "Survived"
        # Return the result and confidence
        return jsonify({'result': result, 'confidence': confidence})
    except Exception as e:
        # Return the error message for debugging
        return jsonify({'error': str(e)})

@app.route('/generate_qr')
def generate_qr():
    url = "http://rtcotomyml.ngrok.app"  # Replace with your website URL
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img_path = os.path.join('static', 'qr_code.png')
    img.save(img_path)
    
    return send_from_directory('static', 'qr_code.png')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use port 8080 or another specific port
    app.run(host='0.0.0.0', port=port, debug=True)