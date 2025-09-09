import os
from chat_bot import ChatBot

def main():
    docs_path = os.path.join(os.path.dirname(__file__), '..', 'docs')

    bot = ChatBot(docx_dir=docs_path)
    bot.run_chat()

if __name__ == '__main__':
    main()