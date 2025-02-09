# FaceSwap_test
FaceSwap with SDM
В данном репозитории представленны 2 реализации алгоритма: оффлайн(с тюном лоры для FLUX1.dev по 5-10 фотографиям) и онлайн(в ComfyUI без тюна)
1 подход)
    Здесь у меня самый удачный вариант получился с помощью обучения LoRA адаптера на человеке для сохранение консистентности и учитвание стиля через redux
    Чтобы затюнить лору использовал ai-toolkit
    ## Installation
    
    Requirements:
    - python >3.10
    - Nvidia GPU with enough ram to do what you need
    - python venv
    - git
    
    
    
    Linux:
    ```bash
    git clone https://github.com/ostris/ai-toolkit.git
    cd ai-toolkit
    git submodule update --init --recursive
    python3 -m venv venv
    source venv/bin/activate
    # .\venv\Scripts\activate on windows
    # install torch first
    pip3 install torch
    pip3 install -r requirements.txt
    ```
    
    Windows:
    ```bash
    git clone https://github.com/ostris/ai-toolkit.git
    cd ai-toolkit
    git submodule update --init --recursive
    python -m venv venv
    .\venv\Scripts\activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```
    ## Training
    Перед обучением лоры запустите код для препрцессинга данных
    ```
    python preproc_data.py <path to folder with images>
    ```
    Дальше нужно изменить в конфиге путь данным  и запустить скрипт
    ```
    python run.py train_lora_flux_24gb.yaml
    ```
    Обученную лору нужно указать в FaceSwapRedux.ipynb
    
    
