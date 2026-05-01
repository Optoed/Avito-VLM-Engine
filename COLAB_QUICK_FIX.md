# Быстрое исправление для Colab

## Проблема: FileNotFoundError - данные не найдены

### Решение: Загрузите данные в Colab

#### Вариант 1: Через Google Drive (рекомендуется)

```python
# Ячейка 1: Подключите Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Ячейка 2: Скопируйте данные из Drive
import shutil
import os

# Путь к вашим данным в Drive
drive_data_path = '/content/drive/MyDrive/classifier_data'  # Измените на ваш путь
local_data_path = '/content/classifier_data'

if os.path.exists(drive_data_path):
    if os.path.exists(local_data_path):
        shutil.rmtree(local_data_path)
    shutil.copytree(drive_data_path, local_data_path)
    print(f"✅ Данные скопированы из Drive в {local_data_path}")
else:
    print(f"❌ Данные не найдены в {drive_data_path}")
    print("Загрузите папку classifier_data в Google Drive")
```

#### Вариант 2: Загрузка архива напрямую

```python
# Ячейка 1: Загрузите архив classifier_data.zip
from google.colab import files
uploaded = files.upload()  # Выберите classifier_data.zip

# Ячейка 2: Распакуйте
import zipfile
import os

for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/')
        print(f"✅ Распаковано: {filename}")
        os.remove(filename)  # Удаляем архив после распаковки
```

#### Вариант 3: Создать структуру вручную (если данных нет)

Если у вас нет данных, нужно сначала запустить `organize_data.py`:

```python
# Ячейка 1: Загрузите метаданные и изображения
from google.colab import drive
drive.mount('/content/drive')

# Скопируйте avito_data_fashion
import shutil
shutil.copytree('/content/drive/MyDrive/avito_data_fashion', '/content/avito_data_fashion')

# Ячейка 2: Организуйте данные
from organize_data import organize_data
organize_data()  # Создаст classifier_data
```

---

## Также: Включите GPU!

```python
# Проверьте GPU
import torch
print(f"CUDA доступен: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU не доступен! Включите в Runtime -> Change runtime type -> GPU")
```

**В Colab:** Runtime → Change runtime type → Hardware accelerator → GPU

---

## Полная последовательность для Colab:

```python
# ЯЧЕЙКА 1: Установка зависимостей
!pip install timm torch torchvision

# ЯЧЕЙКА 2: Установка wheel (если еще не установлен)
!pip install clothing_classifier-1.0.0-py3-none-any.whl

# ЯЧЕЙКА 3: Подключение Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ЯЧЕЙКА 4: Загрузка данных
import shutil
import os

drive_data = '/content/drive/MyDrive/classifier_data'
local_data = '/content/classifier_data'

if os.path.exists(drive_data):
    if os.path.exists(local_data):
        shutil.rmtree(local_data)
    shutil.copytree(drive_data, local_data)
    print("✅ Данные загружены")
else:
    print("❌ Загрузите classifier_data в Google Drive")

# ЯЧЕЙКА 5: Проверка GPU
import torch
print(f"GPU: {torch.cuda.is_available()}")

# ЯЧЕЙКА 6: Запуск обучения
from train_classifier import main
main()
```

---

## Структура данных должна быть:

```
/content/classifier_data/
├── category_mapping.json  ← Этот файл нужен!
├── train/
│   ├── пальто/
│   ├── джинсы/
│   └── ...
├── val/
│   ├── пальто/
│   ├── джинсы/
│   └── ...
└── test/
    ├── пальто/
    ├── джинсы/
    └── ...
```

---

## Если данных нет на Drive:

1. Заархивируйте папку `classifier_data` на локальной машине
2. Загрузите архив в Google Drive
3. Используйте Вариант 2 выше для распаковки
