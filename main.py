from telegram.ext import Updater, InlineQueryHandler, CommandHandler
from telegram.ext.dispatcher import run_async
import telegram
import requests
import re

def get_url():
    contents = requests.get('https://random.dog/woof.json').json()
    url = contents['url']
    return url

def get_image_url():
    allowed_extension = ['jpg','jpeg','png']
    file_extension = ''
    while file_extension not in allowed_extension:
        url = get_url()
        file_extension = re.search("([^.]*)$",url).group(1).lower()
    return url

@run_async
def bop(update, context):
    url = get_image_url()
    chat_id = update.message.chat_id
    print(chat_id)
    context.bot.send_photo(chat_id=chat_id, photo=url)

def main():
    updater = Updater('5158159363:AAEqfpzHx23wGcBrDTYnBXDxTT1Pxoia2Ug', use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('bop',bop))
    updater.start_polling()
    updater.idle()




def main_2():
    bot = telegram.Bot(token='5158159363:AAEqfpzHx23wGcBrDTYnBXDxTT1Pxoia2Ug')

    for i in range(10):
        bot.send_message(chat_id='5176893430', text='Revolution has started up!')



if __name__ == '__main__':
    main()
    #main_2()