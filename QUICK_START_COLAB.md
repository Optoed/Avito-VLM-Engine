# Быстрый старт для Google Colab

## 🎯 Что мы сделали:

1. ✅ Создали `pyproject.toml` - конфигурация для сборки пакета
2. ✅ Добавили поддержку Vision Transformer (ViT) через библиотеку `timm`
3. ✅ Увеличили количество эпох до 50
4. ✅ Создали скрипт для сборки wheel

---

## 📦 Шаг 1: Соберите Wheel на локальной машине

```bash
# Установите инструменты сборки
pip install build wheel

# Соберите wheel
python -m build

# Или используйте скрипт
python build_wheel.py
```

**Результат:** Файл `dist/clothing_classifier-1.0.0-py3-none-any.whl`

---

## 🚀 Шаг 2: Загрузите в Google Colab

### Вариант A: Прямая загрузка
```python
from google.colab import files
uploaded = files.upload()  # Выберите .whl файл
!pip install clothing_classifier-1.0.0-py3-none-any.whl
```

### Вариант B: Через Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/clothing_classifier-1.0.0-py3-none-any.whl .
!pip install clothing_classifier-1.0.0-py3-none-any.whl
```

---

## 📊 Шаг 3: Загрузите данные

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/classifier_data /content/
```

---

## 🎓 Шаг 4: Запустите обучение

```python
import torch
from train_classifier import main

print(f"GPU: {torch.cuda.get_device_name(0)}")
main()
```

---

## 🔧 Что изменилось:

### Vision Transformer
- Используется `vit_base_patch16_224` по умолчанию
- Поддержка через библиотеку `timm`
- Автоматическая разморозка последних блоков

### Больше эпох
- Было: 30 эпох
- Стало: 50 эпох
- Early stopping patience: 7 (было 5)

### Доступные модели:
- `vit_base_patch16_224` - Vision Transformer (по умолчанию)
- `vit_large_patch16_224` - Большой ViT
- `deit_base_distilled_patch16_224` - DeiT (Data-efficient Image Transformer)
- `efficientnet_b0` - EfficientNet
- `resnet50` - ResNet

---

## 💡 Полезные команды Colab:

```python
# Проверить GPU
!nvidia-smi

# Посмотреть использование памяти
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")

# Сохранить модель
!cp models/*.pth /content/drive/MyDrive/
```

---

## ❓ FAQ

**Q: Что такое wheel?**  
A: Wheel (.whl) - это формат распространения Python пакетов. Это архив с вашим кодом.

**Q: Зачем нужен pyproject.toml?**  
A: Это современный стандарт для конфигурации Python проектов. Заменяет setup.py.

**Q: Почему Vision Transformer?**  
A: ViT - современная архитектура, часто показывает лучшие результаты на больших датасетах.

**Q: Можно ли использовать EfficientNet?**  
A: Да! Просто измените в CONFIG: `'model_name': 'efficientnet_b0'`
