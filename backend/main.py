from flask import Flask, jsonify, request
from flask_cors import CORS
import os

#from dotenv import load_dotenv
import requests
import json
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk
# Load environment variables
version = "?api-version=2024-02-01&model-version=2023-04-15"
endpoint = "xx"
key = "xx"
img_map = dict()

app = Flask(__name__)
CORS(app)

# Azure Speech API credentials
AZURE_SPEECH_KEY = "xx"
AZURE_REGION = "eastus"

def speech_to_text(audio_file):
    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
        speech_config.speech_recognition_language = "en-US"
        converted_file = "temp_audio_converted.wav"
        convert_to_wav(audio_file, converted_file)
        audio_input = speechsdk.audio.AudioConfig(filename=converted_file)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    except Exception as e:
        print("Error initializing SpeechRecognizer:", e)
        raise
    result = speech_recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech recognized."
    else:
        return "Speech recognition error."

def convert_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_file, format="wav")

@app.route("/speech-to-text", methods=["POST"])
def process_audio():
    audio_file = request.files["audio"]
    file_path = "temp_audio.wav"
    audio_file.save(file_path)

    # Convert speech to text
    transcription = speech_to_text(file_path)

    # Clean up temp file
    os.remove(file_path)

    return jsonify({"text": transcription})

def get_image_embedding(image):
    with open(image, "rb") as img:
        data = img.read()

    # Vectorize Image API

    vectorize_img_url = endpoint + "retrieval:vectorizeImage" + version

    headers = {
        "Content-type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": key
    }

    try:
        r = requests.post(vectorize_img_url, data=data, headers=headers)

        if r.status_code == 200:
            image_vector = r.json()["vector"]
            print(f"Successfully processed {image}.")
            return image_vector
        else:
            print(f"An error occurred while processing {image}. Error code: {r.status_code}.")

    except Exception as e:
        print(f"An error occurred while processing {image}: {e}")

    return None


def get_text_embedding(prompt):
    text = {'text': prompt}

    # Image retrieval API
    vectorize_txt_url = endpoint + "retrieval:vectorizeText" + version

    headers = {
        'Content-type': 'application/json',
        'Ocp-Apim-Subscription-Key': key
    }

    try:
        r = requests.post(vectorize_txt_url, data=json.dumps(text), headers=headers)

        if r.status_code == 200:
            text_vector = r.json()['vector']
            return text_vector
        else:
            print(f"An error occurred while processing the prompt '{text}'. Error code: {r.status_code}.")

    except Exception as e:
        print(f"An error occurred while processing the prompt '{text}': {e}")

    return None


from numpy import dot
from numpy.linalg import norm

def get_cosine_similarity(vector1, vector2):
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


def generate_image_vals(text_vector):
    img_vectors = []
    for filename in os.listdir("photos"):
        img_filename = "photos/" + filename
        image_vector = img_map[filename[:-4]]
        similarity = get_cosine_similarity(image_vector, text_vector)
        img_vectors.append((img_filename, similarity))
    return img_vectors

def get_most_k_filenames(prompt, k):
    sorted_img = sorted(generate_image_vals(get_text_embedding(prompt)), key=lambda x: x[1], reverse=True)
    result = []
    for img in sorted_img:
        result.append(img[0])
    return result[:k]

def generate_image_map():
    for filename in os.listdir("photos"):
        img_filename = "photos/" + filename
        image_vector = get_image_embedding(img_filename)
        img_map[filename[:-4]]=image_vector
    return


@app.route('/search', methods = ["POST"])
def searchImage():
    data = json.loads(request.data)
    print(data)
    description = data['desc']
    return jsonify({'photos': get_most_k_filenames(description, 3)}) 


@app.route('/photos')
def get_photos():
    folder_path = 'photos'
    if not folder_path:
        return jsonify({'error': 'Folder path not provided'}), 400

    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder path does not exist'}), 404

    photos = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            photos.append(os.path.join(folder_path, file_name))

    return jsonify({'photos': photos})


# pre load all the picture to cache when server start
with app.app_context():
    generate_image_map()

if __name__ == '__main__':
    app.run(debug=True)
    
