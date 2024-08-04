import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import joblib
import numpy as np
import qrcode

app = Flask(__name__)

# Define mappings
sex_mapping = {'Male': 1, 'Female': 2}
prehospital_mapping = {'yes': 1, 'no': 2}
trauma_mapping = {
    'Blunt': 1,
    'Penetrating': 2,
    'Burn': 3,
    'Other/unspecified': 4
}

# Define the correct order of features as per the model generation script
feature_order = [
    'SEX', 'AGEYEARS', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 
    'SBP', 'PULSERATE', 'RESPIRATORYRATE', 'TOTALGCS', 'TEMPERATURE', 
    'PREHOSPITALCARDIACARREST', 'TRAUMATYPE', 'MECHANISM'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/rtcotomyml')
def rtcotomyml():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log input data for debugging
        print("Received input data: ", request.form)

        # Get data from form and map categorical inputs
        input_data_dict = {}
        if 'SEX' in request.form and request.form['SEX']:
            input_data_dict['SEX'] = sex_mapping[request.form['SEX']]
        if 'AGEYEARS' in request.form and request.form['AGEYEARS']:
            input_data_dict['AGEYEARS'] = int(request.form['AGEYEARS'])
        if 'EMSSBP' in request.form and request.form['EMSSBP']:
            input_data_dict['EMSSBP'] = int(request.form['EMSSBP'])
        if 'EMSPULSERATE' in request.form and request.form['EMSPULSERATE']:
            input_data_dict['EMSPULSERATE'] = int(request.form['EMSPULSERATE'])
        if 'EMSRESPIRATORYRATE' in request.form and request.form['EMSRESPIRATORYRATE']:
            input_data_dict['EMSRESPIRATORYRATE'] = int(request.form['EMSRESPIRATORYRATE'])
        if 'EMSTOTALGCS' in request.form and request.form['EMSTOTALGCS']:
            input_data_dict['EMSTOTALGCS'] = int(request.form['EMSTOTALGCS'])
        if 'SBP' in request.form and request.form['SBP']:
            input_data_dict['SBP'] = int(request.form['SBP'])
        if 'PULSERATE' in request.form and request.form['PULSERATE']:
            input_data_dict['PULSERATE'] = int(request.form['PULSERATE'])
        if 'RESPIRATORYRATE' in request.form and request.form['RESPIRATORYRATE']:
            input_data_dict['RESPIRATORYRATE'] = int(request.form['RESPIRATORYRATE'])
        if 'TOTALGCS' in request.form and request.form['TOTALGCS']:
            input_data_dict['TOTALGCS'] = int(request.form['TOTALGCS'])
        if 'TEMPERATURE' in request.form and request.form['TEMPERATURE']:
            input_data_dict['TEMPERATURE'] = float(request.form['TEMPERATURE'])
        if 'PREHOSPITALCARDIACARREST' in request.form and request.form['PREHOSPITALCARDIACARREST']:
            input_data_dict['PREHOSPITALCARDIACARREST'] = prehospital_mapping[request.form['PREHOSPITALCARDIACARREST']]
        if 'TRAUMATYPE' in request.form and request.form['TRAUMATYPE']:
            input_data_dict['TRAUMATYPE'] = trauma_mapping[request.form['TRAUMATYPE']]
        if 'MECHANISM' in request.form and request.form['MECHANISM']:
            input_data_dict['MECHANISM'] = int(request.form['MECHANISM'])

        # Ensure at least 2 features are provided
        if len(input_data_dict) < 2:
            return jsonify({'error': 'Please provide at least 2 features'})

        # Log extracted features for debugging
        print("Extracted features: ", input_data_dict.keys())
        print("Input data: ", input_data_dict.values())

        # Determine the model, data type, and penalty (if any) from the form input
        model_type = request.form['MODEL_TYPE']  # 'xgb' or 'logistic_regression'
        data_type = request.form['DATA_TYPE']  # 'EMS', 'ED', or 'EMS_ED'
        penalty = request.form.get('PENALTY', '')  # Only for logistic_regression, otherwise empty

        # Convert "EMS + ED" to "EMS_ED" for file path construction
        if data_type == "EMS + ED":
            data_type = "EMS_ED"

        # Order the features as per the defined feature_order
        ordered_features = [feature for feature in feature_order if feature in input_data_dict]
        input_data = [input_data_dict[feature] for feature in ordered_features]

        # Construct the file path
        base_path = '/Users/JakeCanfield/Documents/Trauma_Surgery_Research/Python/Thoracotomy_tool/models/'
        if model_type == 'xgb':
            model_filename = f"{base_path}{model_type}_model_{data_type}_{'_'.join(ordered_features)}.pkl"
        else:
            model_filename = f"{base_path}{model_type}_{penalty}_model_{data_type}_{'_'.join(ordered_features)}.pkl"

        # Check if model file exists
        if not os.path.exists(model_filename):
            error_message = f"Model file {model_filename} not found"
            print(error_message)
            return jsonify({'error': error_message})

        # Load the model
        model = joblib.load(model_filename)

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
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message})

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
