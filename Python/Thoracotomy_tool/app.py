import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import joblib
import numpy as np
import qrcode
import pandas as pd

app = Flask(__name__)

# Load the performance metrics
metrics_df = pd.read_csv('models/model_performance_metrics.csv')

# Load the trained models
models = {}
model_files = os.listdir('models')
for model_file in model_files:
    if model_file.endswith('.pkl'):
        model_name = model_file.replace('.pkl', '')
        models[model_name] = joblib.load(os.path.join('models', model_file))

# Define mappings
sex_mapping = {'Male': 1, 'Female': 2}
prehospital_mapping = {'yes': 1, 'no': 2, 'Unknown': 3}
trauma_mapping = {
    'Blunt': 1,
    'Penetrating': 2,
    'Burn': 3,
    'Other/unspecified': 4
}

@app.route('/')
def home():
    return render_template('index.html', models=metrics_df['Model Type'].unique(), feature_sets=metrics_df['Data Type'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form and map categorical inputs
        data_type = request.form['data_type']
        model_type = request.form['model_type']
        
        input_data = {
            'SEX': sex_mapping.get(request.form.get('SEX', ''), 0),
            'AGEYEARS': int(request.form.get('AGEYEARS', 0)),
            'EMSSBP': int(request.form.get('EMSSBP', 0)),
            'EMSPULSERATE': int(request.form.get('EMSPULSERATE', 0)),
            'EMSRESPIRATORYRATE': int(request.form.get('EMSRESPIRATORYRATE', 0)),
            'EMSTOTALGCS': int(request.form.get('EMSTOTALGCS', 0)),
            'PREHOSPITALCARDIACARREST': prehospital_mapping.get(request.form.get('PREHOSPITALCARDIACARREST', ''), 0),
            'TRAUMATYPE': trauma_mapping.get(request.form.get('TRAUMATYPE', ''), 0),
            'MECHANISM': int(request.form.get('MECHANISM', 0))
        }
        
        if data_type in ['ED', 'EMS_ED']:
            input_data.update({
                'SBP': int(request.form.get('SBP', 0)),
                'PULSERATE': int(request.form.get('PULSERATE', 0)),
                'RESPIRATORYRATE': int(request.form.get('RESPIRATORYRATE', 0)),
                'TOTALGCS': int(request.form.get('TOTALGCS', 0)),
                'TEMPERATURE': float(request.form.get('TEMPERATURE', 0.0))
            })
        
        # Filter out zero values
        input_data = {k: v for k, v in input_data.items() if v != 0}
        
        # Get the model based on available features
        features = sorted(input_data.keys())
        model_key = f'{model_type}_model_{data_type}_{"_".join(features)}'
        model = models.get(model_key)
        
        if not model:
            raise ValueError(f"No model found for the given features: {features}")
        
        # Convert the input data to a numpy array
        input_data_array = np.array(list(input_data.values())).reshape(1, -1)
        
        # Make prediction and get probability
        prediction = int(model.predict(input_data_array)[0])
        probability = model.predict_proba(input_data_array)[0]
        confidence = np.max(probability) * 100  # Convert to percentage

        # Get the model performance metrics
        metrics = metrics_df[(metrics_df['Data Type'] == data_type) & (metrics_df['Model Type'] == model_type) & (metrics_df['Features'] == ','.join(features))].iloc[0]
        accuracy = metrics['Accuracy'] * 100
        auroc = metrics['AUROC']
        samples_used = metrics['Samples Used']

        # Convert prediction to a descriptive result
        result = "Deceased" if prediction == 1 else "Survived"

        # Return the result and additional information
        return jsonify({
            'result': result,
            'confidence': confidence,
            'AUROC': auroc,
            'samples_used': samples_used,
            'accuracy': accuracy
        })
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