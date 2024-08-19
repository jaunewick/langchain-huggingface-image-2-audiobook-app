# langchain-huggingface-image-2-audiobooks-app

This project converts images into short, narrated audiobooks using Langchain, Hugging Face, and Streamlit.

## Live Demo
Here’s a quick demo of the app in action :

1. Upload an image.
2. The app extracts the text, generates a story, and converts it to an audiobook.
3. The audiobook is available for playback directly on the app.

[(https://github.com/user-attachments/assets/944e93bd-073a-420b-b688-34c803400109)](https://github.com/user-attachments/assets/944e93bd-073a-420b-b688-34c803400109)

## Features
- **Image-to-text :** Converts images to descriptive text using the BLIP model.
- **Story Generation :** Creates a short story based on the extracted text using Mistral-7B-Instruct-v0.3 model.
- **Text to Speech :** Transforms the generated story into an audiobook using the ESPnet2 TTS model.

## Requirements
- Python 3.8+
- Hugging Face Transformers
- Langchain
- Streamlit
- Requests
- dotenv

## Installation
1. **Clone the repository :**
```bash
git clone https://github.com/jaunewick/langchain-huggingface-image-2-audiobooks-app.git
cd langchain-huggingface-image-2-audiobooks-app
```

2. **Install the required packages using Pipenv :**
```bash
pipenv install
```

3. **Set up your environment variables :**
- Create a .env file in the project root directory.
- Add your Hugging Face API token :
```md
HUGGINGFACEHUB_API_TOKEN=<your_token_here>
```

## Usage
1. **Run the Streamlit app :**
```bash
streamlit run app.py
```

2. **Upload an image :**

- Choose a .png image file to upload

3. **View the results and enjoy! :**
- The extracted text (scenario) will be displayed.
- The generated short story will be shown.
- Listen to the audiobook created from the generated story.
