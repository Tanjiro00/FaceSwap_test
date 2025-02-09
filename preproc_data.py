import os
import sys
from pathlib import Path

def create_text_files_for_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует.")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(folder_path, txt_filename)

            with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                txt_file.write("person")

            print(f"Создан файл: {txt_filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python script.py <путь_к_папке>")
    else:
        folder_path = sys.argv[1]
        create_text_files_for_images(folder_path)