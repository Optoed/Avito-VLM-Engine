# Инструкция по использованию в Google Colab

## Что такое Wheel (.whl)?

**Wheel** - это формат распространения Python пакетов. Это архив с вашим кодом, который можно установить через `pip install`.

### Преимущества:
- ✅ Упаковывает весь код в один файл
- ✅ Легко загрузить в Colab
- ✅ Устанавливается одной командой
- ✅ Сохраняет структуру проекта

---

## Шаг 1: Сборка Wheel на локальной машине

### 1.1 Установите build инструменты:
```bash
pip install build wheel
```

### 1.2 Соберите wheel:
```bash
python -m build
```

Это создаст файл в папке `dist/`:
- `clothing_classifier-1.0.0-py3-none-any.whl`

### 1.3 Альтернативный способ (старый):
```bash
python setup.py bdist_wheel
```

---

## Шаг 2: Загрузка в Google Colab

### 2.1 Загрузите wheel файл:
```python
from google.colab import files
uploaded = files.upload()  # Выберите .whl файл
```

### 2.2 Или загрузите через Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Скопируйте wheel в рабочую директорию
!cp /content/drive/MyDrive/clothing_classifier-1.0.0-py3-none-any.whl .
```

### 2.3 Или используйте прямую ссылку (если загрузили на GitHub Releases):
```python
!wget https://github.com/your-repo/releases/download/v1.0.0/clothing_classifier-1.0.0-py3-none-any.whl
```

---

## Шаг 3: Установка в Colab

```python
!pip install clothing_classifier-1.0.0-py3-none-any.whl
```

---

## Шаг 4: Загрузка данных в Colab

### Вариант 1: Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Скопируйте данные
!cp -r /content/drive/MyDrive/classifier_data /content/
```

### Вариант 2: Загрузка архива
```python
from google.colab import files
uploaded = files.upload()  # Загрузите classifier_data.zip

!unzip classifier_data.zip
```

### Вариант 3: Прямая ссылка
```python
!wget https://your-link.com/classifier_data.zip
!unzip classifier_data.zip
```

---

## Шаг 5: Запуск обучения в Colab

```python
import torch
from train_classifier import main

# Проверяем GPU
print(f"CUDA доступен: {torch.cuda.is_available()}")
print(f"Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Запускаем обучение
main()
```

---

## Шаг 6: Сохранение результатов

```python
# Сохраняем модель в Google Drive
from google.colab import drive
drive.mount('/content/drive')

!cp models/best_model_*.pth /content/drive/MyDrive/
```

---

## Полный пример ноутбука Colab

```python
# Ячейка 1: Установка зависимостей
!pip install timm torch torchvision

# Ячейка 2: Загрузка wheel
from google.colab import files
uploaded = files.upload()  # Выберите .whl файл
!pip install clothing_classifier-1.0.0-py3-none-any.whl

# Ячейка 3: Загрузка данных
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/classifier_data /content/

# Ячейка 4: Обучение
import torch
from train_classifier import main

print(f"GPU: {torch.cuda.get_device_name(0)}")
main()

# Ячейка 5: Сохранение
!cp models/*.pth /content/drive/MyDrive/models/
```

---

## Настройки для Colab

В `train_classifier.py` уже настроено:
- ✅ Автоматическое определение GPU/CPU
- ✅ Vision Transformer через timm
- ✅ 50 эпох обучения
- ✅ Early stopping

---

## Полезные команды Colab

```python
# Проверить GPU
!nvidia-smi

# Посмотреть использование памяти
!free -h

# Мониторинг обучения
import matplotlib.pyplot as plt
# (добавьте визуализацию в train_classifier.py)
```

---

## Troubleshooting

### Проблема: "ModuleNotFoundError: No module named 'timm'"
**Решение:**
```python
!pip install timm
```

### Проблема: "CUDA out of memory"
**Решение:** Уменьшите batch_size в CONFIG:
```python
CONFIG['batch_size'] = 16  # Вместо 32
```

### Проблема: Данные не найдены
**Решение:** Проверьте путь:
```python
import os
print(os.listdir('.'))
print(os.path.exists('classifier_data'))
```
