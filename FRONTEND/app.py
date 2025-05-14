import logging
import traceback
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from g2p_en import G2p
import io
import nltk
import tensorflow as tf
import tempfile
import os
import librosa

# Arpabet to IPA conversion dictionary
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}

# Phoneme index to phoneme mapping (should match your LSTM model's output classes)
# Update this mapping according to your model's training data
PHONEME_INDEX_TO_LABEL = {
    0: 'AA', 1: 'AE', 2: 'AH', 3: 'AO', 4: 'AW', 5: 'AY', 
    6: 'B', 7: 'CH', 8: 'D', 9: 'DH', 10: 'EH', 11: 'ER', 
    12: 'EY', 13: 'F', 14: 'G', 15: 'HH', 16: 'IH', 17: 'IY', 
    18: 'JH', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'NG', 
    24: 'OW', 25: 'OY', 26: 'P', 27: 'R', 28: 'S', 29: 'SH', 
    30: 'T', 31: 'TH', 32: 'UH', 33: 'UW', 34: 'V', 35: 'W', 
    36: 'Y', 37: 'Z', 38: 'ZH', 39: 'SIL'  # SIL = silence
}

# Feature extraction config for LSTM model
LSTM_CONFIG = {
    'sample_rate': 16000,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,  # 10ms at 16kHz
    'window_length': 400  # 25ms at 16kHz
}

# Load the pre-trained phoneme model
def load_phoneme_model(model_path):
    """
    Load a pre-trained neural network model from an H5 file
    
    Args:
        model_path (str): Path to the .h5 model file
    
    Returns:
        tf.keras.Model: Loaded machine learning model
    """
    try:
        # Ensure TensorFlow uses CPU if needed
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Uncomment to force CPU usage
        
        # Load with custom_objects if your model has custom layers
        model = tf.keras.models.load_model(model_path)
        logger.info(f"LSTM model successfully loaded from {model_path}")
        
        # Get model summary for debugging
        model.summary(print_fn=logger.info)
        
        return model
    except Exception as e:
        logger.error(f"Error loading LSTM model: {e}")
        logger.error(traceback.format_exc())
        return None

def extract_audio_features(audio_path, config=LSTM_CONFIG):
    """
    Extract MFCC features from audio for LSTM model input
    
    Args:
        audio_path (str): Path to audio file
        config (dict): Configuration parameters
        
    Returns:
        np.array: MFCC features
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=config['sample_rate'])
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=config['n_mfcc'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            window=np.hamming(config['window_length'])
        )
        
        # Transpose to get time as first dimension
        mfccs = mfccs.T
        
        # Add delta and delta-delta features if needed
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=1)
        
        # Normalize features (if your model was trained with normalization)
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
        
        logger.info(f"Extracted features shape: {features.shape}")
        return features
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        raise

def predict_phonemes_with_lstm(model, audio_features):
    """
    Predict phonemes from audio features using LSTM model
    
    Args:
        model (tf.keras.Model): Loaded LSTM model
        audio_features (np.array): Preprocessed audio features
    
    Returns:
        list: Predicted phoneme sequence
    """
    # Make prediction
    predictions = model.predict(np.expand_dims(audio_features, axis=0))
    
    # Get the most likely phoneme for each time step
    phoneme_indices = np.argmax(predictions[0], axis=1)
    
    # Convert indices to phoneme labels
    phoneme_sequence = [PHONEME_INDEX_TO_LABEL.get(idx, 'UNK') for idx in phoneme_indices]
    
    # Post-process: remove consecutive duplicates and silence
    processed_phonemes = []
    prev_phoneme = None
    for phoneme in phoneme_sequence:
        if phoneme != prev_phoneme and phoneme != 'SIL':
            processed_phonemes.append(phoneme)
        prev_phoneme = phoneme
    
    return processed_phonemes

def arpabet_to_ipa_seq(phonemes):
    """
    Convert Arpabet phoneme sequence to IPA
    """
    return ''.join(ARPABET_TO_IPA.get(p.rstrip('012'), p) for p in phonemes)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
CORS(app)

# Initialize G2p
g2p = G2p()

# Initialize models
wav2vec_model, wav2vec_processor = None, None
lstm_model = None

def init_wav2vec():
    """Initialize and return Wav2Vec 2.0 model and processor"""
    logger.info("Loading Wav2Vec 2.0 model and processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return model, processor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-model', methods=['POST'])
def upload_model():
    """Handle uploading of LSTM model H5 file"""
    global lstm_model
    
    try:
        if 'model' not in request.files:
            return jsonify({"error": "No model file provided"}), 400
            
        model_file = request.files['model']
        
        if not model_file.filename.endswith('.h5'):
            return jsonify({"error": "Invalid model file. Must be .h5 format"}), 400
            
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            model_file.save(temp_file.name)
            model_path = temp_file.name
            
        # Load the model
        lstm_model = load_phoneme_model(model_path)
        
        # Remove temp file after loading
        os.unlink(model_path)
        
        if lstm_model is None:
            return jsonify({"error": "Failed to load model"}), 500
            
        return jsonify({"message": "Model loaded successfully"}), 200
        
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Model upload failed", "details": str(e)}), 500

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """
    Convert WAV file to text and phonemes using either Wav2Vec or LSTM model
    """
    global wav2vec_model, wav2vec_processor, lstm_model
    
    try:
        # Check if model selection specified
        model_choice = request.form.get('model_choice', 'wav2vec')
        
        # Check if using LSTM but model not loaded
        if model_choice == 'lstm' and lstm_model is None:
            return jsonify({
                "error": "LSTM model not loaded", 
                "details": "Please upload an LSTM model first"
            }), 400
        
        # Initialize Wav2Vec model if needed and using wav2vec
        if model_choice == 'wav2vec' and (wav2vec_model is None or wav2vec_processor is None):
            wav2vec_model, wav2vec_processor = init_wav2vec()
        
        # Check if audio file is in the request
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({"error": "No audio file", "details": "Audio file is missing"}), 400
        
        audio_file = request.files['audio']
        
        # Log file details for debugging
        logger.info(f"Received file: {audio_file.filename}")
        logger.info(f"File content type: {audio_file.content_type}")
        logger.info(f"Using model: {model_choice}")
        
        # Validate file type
        if not audio_file.filename.lower().endswith('.wav'):
            logger.error(f"Invalid file type: {audio_file.filename}")
            return jsonify({
                "error": "Invalid file type", 
                "details": "Only WAV files are supported"
            }), 400
        
        # Read the audio file content
        audio_content = audio_file.read()
        
        # Log audio content details
        logger.info(f"Audio file size: {len(audio_content)} bytes")
        
        # Validate audio content
        if len(audio_content) < 1000:
            logger.warning("Audio file too small")
            return jsonify({
                "error": "Audio file too small", 
                "details": f"Expected >1000 bytes, got {len(audio_content)} bytes"
            }), 400
            
        # Save the BytesIO content to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        logger.info(f"Temporary file created at: {temp_file_path}")
        
        # Process based on model choice
        if model_choice == 'lstm':
            # Process with LSTM model
            text = "LSTM Direct Phoneme Recognition"  # LSTM doesn't produce text directly
            
            # Extract features for LSTM
            features = extract_audio_features(temp_file_path)
            
            # Predict phonemes using LSTM
            phonemes = predict_phonemes_with_lstm(lstm_model, features)
            
        else:  # Default to wav2vec
            # Load the audio using torchaudio
            waveform, sample_rate = torchaudio.load(temp_file_path)
            
            # Resample if needed - Wav2Vec 2.0 expects 16kHz
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate} to 16000 Hz")
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                logger.info("Converting stereo to mono")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Process audio with Wav2Vec 2.0
            input_values = wav2vec_processor(waveform.squeeze().numpy(), 
                                           sampling_rate=sample_rate, 
                                           return_tensors="pt").input_values
            
            # Get logits from model
            with torch.no_grad():
                logits = wav2vec_model(input_values).logits
            
            # Get predicted IDs and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            text = wav2vec_processor.batch_decode(predicted_ids)[0]
            
            # Convert text to phonemes using G2p
            phonemes = g2p(text)
        
        # Remove the temporary file after processing
        os.unlink(temp_file_path)
        logger.info(f"Temporary file removed")
        
        # Convert to IPA
        ipa = arpabet_to_ipa_seq(phonemes)
        
        return jsonify({
            "original_text": text,
            "phonemes": phonemes,
            "ipa": ipa,
            "recognition_method": model_choice
        })
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Audio processing failed",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Default model path - will be overridden by uploads
    DEFAULT_MODEL_PATH = 'phoneme.h5'
    
    # Try to pre-load LSTM model if available
    if os.path.exists(DEFAULT_MODEL_PATH):
        lstm_model = load_phoneme_model(DEFAULT_MODEL_PATH)
    else:
        logger.warning(f"Default LSTM model not found at {DEFAULT_MODEL_PATH}")
    
    # Pre-load Wav2Vec model and processor
    wav2vec_model, wav2vec_processor = init_wav2vec()
    
    app.run(debug=True)