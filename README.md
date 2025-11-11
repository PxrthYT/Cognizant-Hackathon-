🌱 Sustainable Farming AI (Hackathon Project)
This project identifies soil type from an uploaded image and provides crop suggestions to help farmers make sustainable decisions and use land efficiently.
📁 Project Structure
|-- data/
|-- venv/
|-- .env
|-- app.py
|-- main.py
|-- test_embed.py
|-- requirements.txt
|-- predictions_log.csv
|-- soil_memory_meta.json
|-- soil_memory_X.npy
🚀 Features
Upload soil image
AI model classifies soil type using embeddings + kNN
Returns the closest match with confidence
Logs all predictions for learning & monitoring
Supports future expansion to AI-generated farming insights
🧠 Tech Used
Python
PyTorch
Streamlit
NumPy / Sklearn
OpenAI/Gemini API (future chart + insights)
▶️ Run the App
1. Install Dependencies
pip install -r requirements.txt
2. Create .env File
Add your API key:
GOOGLE_API_KEY=your_key_here
3. Start Streamlit App
streamlit run app.py
📊 Logs
All predictions are saved to:
predictions_log.csv
❤️ Purpose
This project promotes sustainable farming by helping farmers optimize soil usage instead of clearing forests for new land.
