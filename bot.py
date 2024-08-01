!pip install transformers accelerate torch telegram requests
!pip install diffusers
!pip install openai
!pip install Pillow

import logging
import requests
import time
import os
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering
from diffusers import DiffusionPipeline
import torch

# Initialize the model and tokenizer for translation
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

# Initialize the diffusion pipeline for image generation
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# Initialize the BLIP processor and model for VQA
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Your Telegram bot token (use environment variables for security)
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
BASE_URL = f'https://api.telegram.org/bot{TOKEN}/'

# Initialize logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def get_updates(offset=None):
    """Fetches updates from the Telegram bot API."""
    url = BASE_URL + 'getUpdates'
    params = {'timeout': 100, 'offset': offset}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get updates: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Exception while getting updates: {e}")
        return {}

def send_message(chat_id, text):
    """Sends a message to the specified chat via the Telegram bot API."""
    url = BASE_URL + 'sendMessage'
    data = {'chat_id': chat_id, 'text': text}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            logger.error(f"Failed to send message: {response.status_code}")
    except Exception as e:
        logger.error(f"Exception while sending message: {e}")

def send_photo(chat_id, photo_path):
    """Sends a photo to the specified chat via the Telegram bot API."""
    url = BASE_URL + 'sendPhoto'
    with open(photo_path, 'rb') as photo:
        data = {'chat_id': chat_id}
        files = {'photo': photo}
        try:
            response = requests.post(url, data=data, files=files)
            if response.status_code != 200:
                logger.error(f"Failed to send photo: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception while sending photo: {e}")

def download_photo(file_id):
    """Downloads a photo from the Telegram server."""
    url = BASE_URL + f'getFile?file_id={file_id}'
    response = requests.get(url)
    if response.status_code == 200:
        file_path = response.json()['result']['file_path']
        photo_url = f'https://api.telegram.org/file/bot{TOKEN}/{file_path}'
        photo = requests.get(photo_url, stream=True)
        if photo.status_code == 200:
            photo_path = f'{file_id}.jpg'
            with open(photo_path, 'wb') as f:
                for chunk in photo.iter_content(1024):
                    f.write(chunk)
            return photo_path
        else:
            logger.error(f"Failed to download photo: {photo.status_code}")
    else:
        logger.error(f"Failed to get file info: {response.status_code}")
    return None

def process_message(message):
    """Processes incoming messages and triggers appropriate actions."""
    text = message.get('text')
    chat_id = message['chat']['id']

    if text:
        if text.lower().startswith("translate"):
            # Handle specific language translations
            if "to Telugu" in text:
                input_text = text.replace("translate", "").strip()
                input_ids = tokenizer(f"translate English to Telugu: {input_text}", return_tensors="pt").input_ids.to("cuda")
            else:
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

            outputs = model.generate(input_ids)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            send_message(chat_id, translated_text)
        elif text.lower().startswith("generate image"):
            prompt = text[len("generate image"):].strip()
            send_message(chat_id, 'Generating image, please wait...')
            image = pipe(prompt=prompt).images[0]
            image_path = 'generated_image.png'
            image.save(image_path)
            send_photo(chat_id, image_path)
        else:
            send_message(chat_id, 'Please send a message in one of the following formats:\n'
                                  '"translate [language1] to [language2]: [text]"\n'
                                  '"generate image [description]"')

    if 'photo' in message:
        photo_file_id = message['photo'][-1]['file_id']
        photo_path = download_photo(photo_file_id)
        if photo_path:
            if text:
                question = text
                raw_image = Image.open(photo_path).convert('RGB')
                inputs = blip_processor(raw_image, question, return_tensors="pt")
                out = blip_model.generate(**inputs)
                answer = blip_processor.decode(out[0], skip_special_tokens=True)
                send_message(chat_id, answer)
            else:
                send_message(chat_id, 'Please provide a question along with the photo.')

def main():
    """Main function to run the Telegram bot."""
    offset = None

    while True:
        try:
            updates = get_updates(offset)
            if 'result' in updates:
                for update in updates['result']:
                    if 'message' in update:
                        process_message(update['message'])
                        offset = update['update_id'] + 1
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(5)

if __name__ == '__main__':
    main()
