
# RTP Voice Predictor 🎙️

The **RTP Voice Predictor** is an AI-based system designed to analyze Real Time Team Performance audio streams and predict key voice-related metrics such as speech quality, speaker emotion, or packet loss impact using Machine Learning.

## 🚀 Features

- 🔊 Processes RTP audio packets in real-time or from capture files 
- 🤖 Predicts voice quality or speaker traits using trained ML models
- 📊 Visualizes predictions and audio insights
- 🧠 Supports model training and evaluation
- 🧪 Testable with synthetic or captured RTP streams

## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** scikit-learn, TensorFlow/Keras, PyTorch (based on implementation)
- **Audio Processing:** mfcc, librosa, wave
- **Packet Parsing:** pyshark
- **Deployment (Optional):** Flask 



## 🧪 How to Use

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run prediction**
   ```bash
    python app.py runserver
   ```


## 🔮 Model Capabilities

- Supports multi-label output such as:
  - Speech clarity score
  - Emotion prediction (angry, happy, neutral, etc.)
  - Noise level estimation

## 📈 Future Enhancements

- Real-time WebRTC integration
- Multi-speaker voice separation
- REST API for third-party integration
- Dashboard for live monitoring

