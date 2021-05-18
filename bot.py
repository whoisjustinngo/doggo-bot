import configparser as cfg
from telegram import *
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import requests
import json

PORT = int(os.environ.get('PORT', 5000))

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def read_token_from_config_file(config):
    parser = cfg.ConfigParser()
    parser.read(config)
    return parser.get('creds', 'token')

def start(update, context):
    message = "Send a photo of a dog to this bot and it will tell you what breed it is!!"
    update.message.reply_text(message)

def info(update, context):
    message = "Send an image of a dog!\nIn the meantime, here's one:"
    url = get_url()
    context.bot.send_message(chat_id=update.effective_chat.id, text = message)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo= url)

def get_url():
    contents = requests.get('https://random.dog/woof.json').json()    
    url = contents['url']
    return url

def recognise(update, context):
    file_id = update.message.photo[-1].file_id
    chat_id = update.message.chat_id
    # gets numpy array of image
    array = get_array(update, file_id, chat_id)
    # update.message.reply_text(str(array))
    result = make_prediction(array)
    update.message.reply_text("This is a " + result)

def get_array(update, file_id, chat_id):
    filepath = os.path.expanduser('~') + '/' + file_id
    bot.send_message(chat_id=chat_id, text="Identifying...")
    bot.send_chat_action(chat_id=chat_id, action=telegram.ChatAction.TYPING)
    file = bot.get_file(file_id).download(filepath)
    array = cv2.imread(file)
    return array

def make_prediction(array):
    breeds = {36: 'Yorkshire terrier', 118: 'dhole', 46: 'giant schnauzer', 103: 'Leonberg', 113: 'toy poodle', 116: 'Mexican hairless', 70: 'Irish water spaniel', 9: 'Afghan hound', 13: 'bluetick', 95: 'Great Dane', 115: 'standard poodle', 89: 'Appenzeller', 82: 'Bouvier des Flandres', 106: 'Samoyed', 44: 'Boston bull', 101: 'basenji', 99: 'Siberian husky', 83: 'Rottweiler', 79: 'Shetland sheepdog', 12: 'bloodhound', 21: 'whippet', 91: 'boxer', 29: 'American Staffordshire terrier', 32: 'Kerry blue terrier', 75: 'briard', 93: 'Tibetan mastiff', 25: 'Saluki', 63: 'Gordon setter', 8: 'Rhodesian ridgeback', 94: 'French bulldog', 111: 'Pembroke', 5: 'Blenheim spaniel', 67: 'Welsh springer spaniel', 59: 'haired pointer', 54: 'coated retriever', 48: 'Scotch terrier', 85: 'Doberman', 96: 'Saint Bernard', 100: 'affenpinscher', 15: 'Walker hound', 43: 'Dandie Dinmont', 47: 'standard schnauzer', 34: 'Norfolk terrier', 23: 'Norwegian elkhound', 11: 'beagle', 62: 'Irish setter', 98: 'malamute', 40: 'Airedale', 19: 'Irish wolfhound', 31: 'Border terrier', 72: 'schipperke', 68: 'cocker spaniel', 61: 'English setter', 76: 'kelpie', 30: 'Bedlington terrier', 81: 'Border collie', 50: 'silky terrier', 78: 'Old English sheepdog', 27: 'Weimaraner', 60: 'vizsla', 74: 'malinois', 107: 'Pomeranian', 49: 'Tibetan terrier', 16: 'English foxhound', 80: 'collie', 64: 'Brittany spaniel', 119: 'African hunting dog', 105: 'Great Pyrenees', 84: 'German shepherd', 20: 'Italian greyhound', 51: 'coated wheaten terrier', 73: 'groenendael', 110: 'Brabancon griffon', 0: 'Chihuahua', 24: 'otterhound', 55: 'coated retriever', 57: 'Labrador retriever', 37: 'haired fox terrier', 108: 'chow', 90: 'EntleBucher', 66: 'English springer', 38: 'Lakeland terrier', 1: 'Japanese spaniel', 69: 'Sussex spaniel', 42: 'Australian terrier', 33: 'Irish terrier', 6: 'papillon', 52: 'West Highland white terrier', 26: 'Scottish deerhound', 28: 'Staffordshire bullterrier', 86: 'miniature pinscher', 77: 'komondor', 17: 'redbone', 109: 'keeshond', 41: 'cairn', 58: 'Chesapeake Bay retriever', 2: 'Maltese dog', 18: 'borzoi', 35: 'Norwich terrier', 53: 'Lhasa', 112: 'Cardigan', 97: 'Eskimo dog', 117: 'dingo', 104: 'Newfoundland', 88: 'Bernese mountain dog', 10: 'basset', 71: 'kuvasz', 65: 'clumber', 114: 'miniature poodle', 39: 'Sealyham terrier', 87: 'Greater Swiss Mountain dog', 7: 'toy terrier', 3: 'Pekinese', 45: 'miniature schnauzer', 4: 'Tzu', 22: 'Ibizan hound', 14: 'tan coonhound', 102: 'pug', 56: 'golden retriever', 92: 'bull mastiff'}
    # resize array
    array = tf.image.resize(array, (224, 224))
    array = tf.reshape(array, [1, 224, 224, 3])
    p = model.predict(array)
    prediction = breeds[p.argmax()]
    return prediction

def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)
    
def main():
    # initialise bot
    global bot
    global updater
    bot = telegram.Bot(token= read_token_from_config_file("config.cfg"))
    updater = Updater(token=read_token_from_config_file("config.cfg"), use_context= True)
    dispatcher = updater.dispatcher

    # add handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo, recognise)) # handler to send to bot if image is detected
    dispatcher.add_handler(MessageHandler(Filters.all, info))
    dispatcher.add_error_handler(error)

    # establish comnnection with the server
    # if any messages sent to server and is image, send the image to the model to classify
    # format the returened data and send it back to the user

    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=read_token_from_config_file("config.cfg"))
    updater.bot.setWebhook('<your heroku url>' + read_token_from_config_file("config.cfg"))
    updater.idle()

if __name__ == "__main__":
    # load trained model
    model = tf.keras.models.load_model("./")
    main()
