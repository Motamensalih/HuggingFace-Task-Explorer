import streamlit as st
from transformers import pipeline
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import gc
import inflect

import io
import matplotlib.pyplot as plt

from transformers import BlipForImageTextRetrieval
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration
from transformers import BlipForQuestionAnswering

from transformers.utils import logging
logging.set_verbosity_error()


# Define tasks and corresponding models
# Future update: to include more models per task and let the user choose one
# Could be used to compare responses of different models
TASKS = {
    # 'Text Generation': ['openai-community/gpt2'],
    'Translation':['facebook/nllb-200-distilled-600M'],
    'Summarization': ['facebook/bart-large-cnn'],
    'Object Detection':['facebook/detr-resnet-50'],
    # 'Image Segmentation':['Zigeng/SlimSAM-uniform-77'],
    'Image Retrieval':['Salesforce/blip-itm-base-coco'],
    'Image Captioning':['Salesforce/blip-image-captioning-base'],
    'Visual Question Answering':['Salesforce/blip-vqa-base"']
}

# Dsiplay required input field based on the choosen task
def handle_input(task):
    if task in ['Summarization', 'Translation']: # Text Generation was excluded
        user_input = st.text_area('Enter your text:')
        if task == 'Translation':
            target_language = st.selectbox('Select Target Language:', ['Arabic', 'French', 'German'])
            return user_input, target_language
        return user_input, None

    elif task in ['Image Segmentation', 'Image Captioning', 'Object Detection', 'Visual Question Answering', 'Image Retrieval']:
        image_path = uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if image_path:
          image = Image.open(image_path)
          st.image(image, caption='Uploaded Image.', use_container_width=True)
          st.write("Image successfully uploaded!")
          if task in ['Visual Question Answering', 'Image Retrieval']:
            text =  st.text_area('Enter your text:')
            return image, text
          return image, None
        return None, None

# Draw bounding boxes on the detected objectes (for Object Detection task)
def render_results_in_image(in_pil_img, in_results):
    plt.figure(figsize=(16, 10))
    plt.imshow(in_pil_img)

    ax = plt.gca()

    for prediction in in_results:

        x, y = prediction['box']['xmin'], prediction['box']['ymin']
        w = prediction['box']['xmax'] - prediction['box']['xmin']
        h = prediction['box']['ymax'] - prediction['box']['ymin']

        ax.add_patch(plt.Rectangle((x, y),
                                   w,
                                   h,
                                   fill=False,
                                   color="green",
                                   linewidth=2))
        ax.text(
           x,
           y,
           f"{prediction['label']}: {round(prediction['score']*100, 1)}%",
           color='red'
        )

    plt.axis("off")

    # Save the modified image to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png',
                bbox_inches='tight',
                pad_inches=0)
    img_buf.seek(0)
    modified_image = Image.open(img_buf)

    # Close the plot to prevent it from being displayed
    plt.close()

    return modified_image

# Excluded: Color-filling the segments (for Image Segmentation task)
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3),
#                                 np.array([0.6])],
#                                axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# def show_pipe_masks_on_image(raw_image, outputs):
#     plt.imshow(np.array(raw_image))
#     ax = plt.gca()
#     for mask in outputs["masks"]:
#         show_mask(mask, ax=ax, random_color=True)
#     plt.axis("off")
#     # Save the modified image to a BytesIO object
#     img_buf = io.BytesIO()
#     plt.savefig(img_buf, format='png',
#                 bbox_inches='tight',
#                 pad_inches=0)
#     img_buf.seek(0)
#     modified_image = Image.open(img_buf)

#     # Close the plot to prevent it from being displayed
#     plt.close()

#     return modified_image
#     # plt.show()

# Count number of each detected object (for Object Detection task)
def summarize_predictions_natural_language(predictions):
    summary = {}
    p = inflect.engine()

    for prediction in predictions:
        label = prediction['label']
        if label in summary:
            summary[label] += 1
        else:
            summary[label] = 1

    result_string = "In this image, there are "
    for i, (label, count) in enumerate(summary.items()):
        count_string = p.number_to_words(count)
        result_string += f"{count_string} {label}"
        if count > 1:
          result_string += "s"

        result_string += ", "

        if i == len(summary) - 2:
          result_string += " and "

    # Remove the trailing comma and space
    result_string = result_string.rstrip(', ') + "."

    return result_string

# Streamlit App
st.set_page_config(page_title='HuggingFace Task Explorer', page_icon='🤗')
st.sidebar.header('HuggingFace Task Explorer')
task = st.sidebar.selectbox('Select a Task:', list(TASKS.keys()))
model = st.sidebar.selectbox('Select a Model:', TASKS[task])
st.title(task)

# Future update
# model_name = st.sidebar.selectbox('Select a Model:', TASKS[task])
user_input, extra_param = handle_input(task)

# Generate response after pressing the buttton
if st.button('Generate'):
    if user_input:
      # if task == 'Text Generation':
      #   model = pipeline("text-generation", model="openai-community/gpt2")
      #   output = model(user_input)
      #   st.write('Model Output:', output)

      if task == 'Translation':
        model = pipeline(task="translation",
                      model="facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.bfloat16)
        lang_code_mapping = {'Arabic':'arz_Arab'   , 'French':'fra_Latn'   , 'German':'deu_Latn'}
        text_translated = model(user_input,
                             src_lang="eng_Latn",
                             tgt_lang=lang_code_mapping[extra_param])
        output = text_translated[0]['translation_text']
        st.write('Translated Text:', output)
      
      elif task == 'Summarization':
        model = pipeline(task="summarization",
                      model="facebook/bart-large-cnn",
                      torch_dtype=torch.bfloat16)
        summary = model(user_input,
                     min_length=10,
                     max_length=100)
        output = summary[0]['summary_text']
        st.write('Summarized Text:', output)

      elif task == 'Object Detection':
        model = pipeline("object-detection", "facebook/detr-resnet-50")
        #user_input.resize((569, 491))
        pipeline_output = model(user_input)
        processed_image = render_results_in_image(
                                                  user_input, 
                                                  pipeline_output)
        output_text = summarize_predictions_natural_language(pipeline_output)
        st.image(processed_image, caption="Processed Image")
        st.write('Model Output:', output_text)
      
      # elif task == 'Image Segmentation':
      #   model = pipeline("mask-generation", "Zigeng/SlimSAM-uniform-77")
      #   #user_input.resize((720, 375))
      #   output = model(user_input, points_per_batch=32)
      #   processed_image = show_pipe_masks_on_image(user_input, output)
      #   st.image(processed_image, caption="Processed Image")

      elif task == 'Image Captioning':
        model = BlipForConditionalGeneration.from_pretrained( "Salesforce/blip-image-captioning-base")
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(user_input, return_tensors="pt")
        out = model.generate(**inputs)
        st.write('Model Output:', processor.decode(out[0], skip_special_tokens=True))

      elif task == 'Image Retrieval' and extra_param:
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        inputs = processor(images=user_input,
                           text=extra_param,
                           return_tensors="pt")
        itm_scores = model(**inputs)[0]
        itm_score = torch.nn.functional.softmax(itm_scores,dim=1)
        st.write('Model Output:',f"The image and text are matched with a probability of {itm_score[0][1]:.4f}")

      elif task == 'Visual Question Answering' and extra_param:
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        inputs = processor(user_input, extra_param, return_tensors="pt")
        out = model.generate(**inputs)
        st.write('Model Output:',processor.decode(out[0], skip_special_tokens=True))
      del model
      gc.collect()
    else:
        st.warning('Please provide the necessary input.')
