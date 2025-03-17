# FaceSwap_test  
**FaceSwap with SDM**  

В данном репозитории представлены две реализации алгоритма замены лиц:  
- **Оффлайн**: Тюн LoRA адаптера для FLUX1.dev на основе 5-10 фотографий + Redux + Realism LoRA показал самый лучший результат.  
- **Онлайн**: В ComfyUI развернул workflow c ReActor + CodeFormer. Так же предоставил аналогичный пайплайн в .ipynb формате

---

## **1. Оффлайн подход**  
Самый удачный вариант был достигнут с помощью обучения LoRA адаптера на фотографиях человека. Это позволяет сохранить консистентность и учитывать стиль через Redux.  

### **Установка** 

#### **Linux**  
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
# Установите torch первым
pip3 install torch
pip3 install -r requirements.txt
```

#### **Windows**  
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### **Обучение LoRA**  

1. **Препроцессинг данных**  
   Перед обучением LoRA выполните предварительную обработку данных:  
   ```bash
   python preproc_data.py <path to folder with images>
   ```

2. **Настройка конфигурации**  
   Измените путь к данным в конфиге train_lora_flux_24gb.yaml.  

3. **Запуск обучения**  
   Запустите скрипт обучения:  
   ```bash
   python run.py train_lora_flux_24gb.yaml
   ```

4. **Использование обученной LoRA**  
   Укажите путь к обученной LoRA в `FaceSwapRedux.ipynb`, предворительно установив зависимости из requirements.txt.
   
Веса лоры обученной на данных из ноушена лежат тут https://huggingface.co/Deenchik/faceLora/blob/main/face_lora.safetensors
#### **Результаты**
![out1](https://github.com/user-attachments/assets/a0930f45-400e-415b-b6be-6768f7f4381f)

---

## **2. Онлайн подход**  
Для онлайн-реализации используется чистый ReActor . 
Так же тестировал Face ID + Style IP-Adapter + ControlNet++, он показал плохой результат
Весь пайплайн находиться в ноутбуке SDXL_Face.ipynb

---


---

