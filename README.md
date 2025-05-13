# HuggingFace Task Explorer

This is a Streamlit-based application that provides an interactive interface for exploring various HuggingFace models. It supports tasks like Translation, Summarization, Object Detection, Image Retrieval, Image Captioning, and Visual Question Answering. Users can select a task, upload images or provide text, and get processed results using pre-trained models from HuggingFace.

## Features

* **Translation**: Supports translation from English to Arabic, French, and German.
* **Summarization**: Generates concise summaries for long texts.
* **Object Detection**: Identifies objects in an image and displays bounding boxes with labels.
* **Image Retrieval**: Matches images with a provided text prompt.
* **Image Captioning**: Generates descriptive captions for uploaded images.
* **Visual Question Answering**: Answers questions based on the content of an uploaded image.

## Technologies Used

* **Streamlit** for interactive UI
* **HuggingFace Transformers** for state-of-the-art NLP and CV models
* **Pillow** for image processing
* **Matplotlib** for rendering object detection results
* **PyTorch** for model execution

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Motamensalih/HuggingFace-Task-Explorer.git
   cd HuggingFace-Task-Explorer
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

5. Access the app in your browser by pressing the provided URL. It looks something similar to this:

   [http://localhost:8501](http://localhost:8501)

## Dependencies

* `transformers`
* `streamlit`
* `Pillow`
* `matplotlib`
* `torch`
* `inflect`

Install them with:

```bash
pip install transformers streamlit Pillow matplotlib torch inflect
```

## Usage

1. Choose a task from the sidebar.
2. Provide the required input (text or image).
3. Click **Generate** to see the model's output.

### Supported Tasks:

* **Translation** → Enter text and select a target language.
* **Summarization** → Enter long text to receive a summary.
* **Object Detection** → Upload an image to detect objects.
* **Image Retrieval** → Upload an image and provide a query to match.
* **Image Captioning** → Upload an image to generate a caption.
* **Visual Question Answering** → Upload an image and ask a question.

## Future Improvements

* Add more models per task and allow users to compare results.
* Support for Image Segmentation and Text Generation.
* Include more tasks - speech to text and text to speech.
* Creating interactions between different tasks - combining CV models with text-to-speech models to help visually-impaired people to understand the content of the images.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* HuggingFace for providing pre-trained models.
* Streamlit for the interactive web application framework.

Feel free to contribute to this project or suggest improvements!

