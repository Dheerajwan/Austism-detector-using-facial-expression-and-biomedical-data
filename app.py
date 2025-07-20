import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Ensure the 'static/uploads' directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained autism detection model
try:
    model = load_model('autism (1).h5')
    print("AI model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(img_path):
    """
    Preprocesses the uploaded image for model prediction.
    """
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/', methods=['GET'])
def index_page():
    """
    Renders the main HTML page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image and biomedical data for autism prediction.
    Ensures all error responses are JSON.
    """
    if not model:
        print("Error: Model not loaded when /predict was called.")
        return jsonify({'error': 'AI model not loaded on server.'}), 500

    image_result = None
    biomedical_result = None
    img_path = None

    try:
        # Process image if uploaded
        if 'image' in request.files and request.files['image'].filename != '':
            img_file = request.files['image']
            filename = secure_filename(img_file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(img_path)
            
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)[0][0]

            if prediction > 0.5:
                image_result = 'Non Autistic'
            else:
                image_result = 'Autistic'
            print(f"Image prediction: {image_result} (Score: {prediction})")

        # Process biomedical data if provided
        biomedical_data_str = request.form.get('biomedicalData')
        if biomedical_data_str:
            try:
                biomedical_params = json.loads(biomedical_data_str)
                print(f"Received biomedical data: {biomedical_params}")
                
                eeg_str = biomedical_params.get('eeg', '')
                eeg_score = float(eeg_str) if eeg_str != '' else 0.0

                heart_rate_str = biomedical_params.get('heartRate', '')
                heart_rate = float(heart_rate_str) if heart_rate_str != '' else 0.0

                cholesterol_str = biomedical_params.get('cholesterol', '')
                cholesterol = float(cholesterol_str) if cholesterol_str != '' else 0.0
                
                if eeg_score > 7.0 and heart_rate < 70.0:
                    biomedical_result = 'Potential Autism Risk (based on biomedical data)'
                else:
                    biomedical_result = 'Low Autism Risk (based on biomedical data)'
                print(f"Biomedical prediction: {biomedical_result}")

            except json.JSONDecodeError:
                print(f"JSONDecodeError: Invalid JSON format for biomedical data: {biomedical_data_str}")
                return jsonify({'error': 'Invalid JSON format for biomedical data. Please check your inputs.'}), 400
            except ValueError as ve:
                print(f"ValueError: Could not convert biomedical data to number: {ve}")
                return jsonify({'error': f'Invalid biomedical data format. Please ensure numerical inputs are numbers: {ve}'}), 400
            except Exception as e:
                print(f"Error processing biomedical data: {e}")
                return jsonify({'error': f"Error processing biomedical data: {e}"}), 400

        # Combine predictions (placeholder logic)
        final_prediction = "Unable to determine"
        if image_result and biomedical_result:
            if image_result == 'Autistic' and 'Autism Risk' in biomedical_result:
                final_prediction = 'Highly Suggestive of Autism'
            elif image_result == 'Autistic' or 'Autism Risk' in biomedical_result:
                final_prediction = 'Suggestive of Autism (consider more tests)'
            else:
                final_prediction = 'Not Autistic'
        elif image_result:
            final_prediction = f'Image Prediction: {image_result}'
        elif biomedical_result:
            final_prediction = f'Biomedical Prediction: {biomedical_result}'
        else:
            final_prediction = 'No data provided for prediction.'
        
        print(f"Combined prediction: {final_prediction}")

        return jsonify({
            'image_prediction': image_result,
            'biomedical_prediction': biomedical_result,
            'combined_prediction': final_prediction,
            'image_path': '/' + img_path if img_path else None
        })

    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500

@app.route('/healthy-tips', methods=['GET'])
def healthy_tips():
    tips = [
        "**Early Intervention:** Early diagnosis and intervention can significantly improve outcomes. Therapies like Applied Behavior Analysis (ABA), speech therapy, and occupational therapy can be very beneficial.",
        "**Structured Environment:** Creating a predictable and structured environment can help individuals with autism feel secure and reduce anxiety. Visual schedules and clear routines are often helpful.",
        "**Communication Strategies:** Explore various communication methods, including verbal communication, picture exchange systems (PECS), or augmentative and alternative communication (AAC) devices, to find what works best.",
        "**Sensory Regulation:** Many individuals with autism have sensory sensitivities. Identifying and addressing these sensitivities through sensory diets, calming spaces, or sensory-friendly activities can improve comfort and focus.",
        "**Diet and Nutrition:** While there's no specific 'autism diet,' some families find that dietary interventions (e.g., gluten-free, casein-free diets) can help with certain symptoms. Consult with a healthcare professional before making significant dietary changes.",
        "**Physical Activity:** Regular physical activity can help with motor skills, sensory integration, and overall well-being. Activities like swimming, yoga, or martial arts can be particularly beneficial.",
        "**Support Networks:** Connect with other families, support groups, and organizations dedicated to autism. Sharing experiences and resources can be invaluable.",
        "**Individualized Education Programs (IEPs):** For children, an individualized education program (IEP) tailored to their specific needs can provide necessary academic and developmental support in school.",
        "**Self-Care for Caregivers:** Caring for an individual with autism can be demanding. Prioritizing self-care, seeking respite, and maintaining personal well-being are crucial for caregivers."
    ]
    return jsonify({'tips': tips})

@app.route('/creator-details', methods=['GET'])
def creator_details():
    """
    Provides project creator details.
    """
    details = {
        "Project Name": "Autism Spectrum Disorder Detector",
        "Creators": [
            {"Name": "Dheeraj Wan", "Role": "Group Leader", "Image": "https://placehold.co/100x100?text=Dheeraj"},
            {"Name": "Sonil Talreja", "Role": "Team Member", "Image": "https://placehold.co/100x100?text=Sonil"},
            {"Name": "Sahil Chhabria", "Role": "Team Member", "Image": "https://placehold.co/100x100?text=Sahil"}
        ],
        "Contact": "asdetectectco@gmail.com", # Replace with actual contact info if available
        "Version": "1.0.0",
        "Date": "July 2025"
    }
    return jsonify({'details': details})

if __name__ == '__main__':
    app.run(debug=True)