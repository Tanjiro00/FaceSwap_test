# FaceSwap_test  
**FaceSwap with SDM**  

В данном репозитории представлены две реализации алгоритма замены лиц:  
- **Оффлайн**: С использованием тонкой настройки LoRA адаптера для FLUX1.dev на основе 5-10 фотографий + Redux + Realism LoRA.  
- **Онлайн**: В ComfyUI я собирался протестировать идею с ReActor + FLUX + PuLID + Refiner(скорее всего здесь еще нужен ContolNet), но ComfyUI на моей машинке не смог определить кастомные ноды. Поэтому я попробовал использовать в ноутбуке детекциию лиц + ReActor(добавлял Codeformer, но он показал себя плохо). И финальная идея это FaceDetection + Face ID + ControlNet + Florence

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

---

## **2. Онлайн подход**  
Для онлайн-реализации используется **ComfyUI** .  

---


---

