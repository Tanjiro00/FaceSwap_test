# FaceSwap_test  
**FaceSwap with SDM**  

В данном репозитории представлено несколько реализаций алгоритмов замены лиц:  
- **Оффлайн(Тюн LoRA адаптера для FLUX1.dev на основе 5-10 фотографий)**:
- 1) Redux + Realism LoRA. В флоу ComfyUI находиться пайплайн с Depth ControlNet.
- 2) Inpaint c зашумлением 40 процентов (Самый лучшее решение 
- **Онлайн**: В ComfyUI
- 1)PuLID/InstanceID + Redux + Depth/Canny ControlNet
- 2)PulID + Face Segmemtation + Inpaint(самый лучший результат для онлайн решения)
- 2) развернул workflow c ReActor + CodeFormer. Так же предоставил аналогичный пайплайн в .ipynb формате

---

## **1. Оффлайн подход**  
Самый удачный вариант был достигнут с помощью обучения LoRA адаптера на фотографиях человека. Это позволяет сохранить консистентность и заменить лицо через FLUX1-dev Inpainting по маске.  
Для запуска флоу нужно установить ComfyUI(так же есть решение в виде блокнота, но в комфи решение должно работать лучше + удобно интегрировать в сервисы)
Далее обучаем лору через Ai-toolkit
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
![image](https://github.com/user-attachments/assets/aa232065-f2e7-4649-8ef8-46469055034b)
![image](https://github.com/user-attachments/assets/894999e6-8499-4551-aa3d-c4dd387e23dc)
![image](https://github.com/user-attachments/assets/7f721ec2-d45d-4e34-870f-a4e1bf168c75)



---

## **2. Онлайн подход**  
Для онлайн-реализации используется тот же пайплайн что и для оффлайн, только с заменой лоры на PulID
Нужно будет скачать ComfyUI и установить все кастомные ноды
#### **Результаты**
- ![image](https://github.com/user-attachments/assets/a1a10653-ff26-4db4-9817-56ca4aeb1cbd)  ![image](https://github.com/user-attachments/assets/aad6fbb0-0bcb-400d-865a-62b95128d2c5)
- ![image](https://github.com/user-attachments/assets/8a1ff363-4705-4a31-9b2d-50e1c348d588) ![image](https://github.com/user-attachments/assets/520bf440-a2f2-4ac0-9ba5-6cf9efc77c70)
- ![image](https://github.com/user-attachments/assets/623f0d57-7c40-40f1-92f6-21a9df8ea8ba) ![image](https://github.com/user-attachments/assets/75d7ab13-55fc-41d3-908f-668d63d3c05c)





---

![image](https://github.com/user-attachments/assets/5eed03b9-1820-4780-96cc-33e49784a61f)

---

