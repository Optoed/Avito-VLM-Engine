# Объяснение: Wheel, pyproject.toml и Google Colab

## 🎯 Что такое Wheel (.whl)?

**Wheel** - это формат распространения Python пакетов (как ZIP архив, но специально для Python).

### Аналогия:
- **Wheel** = `.exe` файл для Windows
- **Wheel** = `.dmg` для macOS
- **Wheel** = `.deb` для Debian

### Зачем нужен?
1. ✅ Упаковывает весь код в один файл
2. ✅ Легко установить: `pip install package.whl`
3. ✅ Сохраняет структуру проекта
4. ✅ Можно загрузить в Colab одной командой

### Пример:
```
Ваш проект:
  ├─ train_classifier.py
  ├─ organize_data.py
  └─ другие файлы...

После сборки:
  └─ clothing_classifier-1.0.0-py3-none-any.whl  (один файл!)
```

---

## 📝 Что такое pyproject.toml?

**pyproject.toml** - современный файл конфигурации Python проектов.

### Зачем нужен?
- ✅ Заменяет старый `setup.py`
- ✅ Стандарт Python (PEP 518)
- ✅ Проще и понятнее
- ✅ Нужен для сборки wheel

### Что в нем?
```toml
[project]
name = "clothing-classifier"  # Имя пакета
version = "1.0.0"            # Версия
dependencies = [...]          # Зависимости
```

---

## 🚀 Как собрать Wheel?

### Способ 1: Через build (рекомендуется)
```bash
pip install build wheel
python -m build
```

### Способ 2: Через скрипт
```bash
python build_wheel.py
```

### Результат:
```
dist/
  └─ clothing_classifier-1.0.0-py3-none-any.whl
```

---

## 📦 Как использовать в Google Colab?

### Шаг 1: Загрузите wheel
```python
from google.colab import files
uploaded = files.upload()  # Выберите .whl файл
```

### Шаг 2: Установите
```python
!pip install clothing_classifier-1.0.0-py3-none-any.whl
```

### Шаг 3: Используйте
```python
from train_classifier import main
main()
```

---

## 🔄 Полный процесс:

```
1. Локально:
   ├─ Создаете pyproject.toml
   ├─ Собираете wheel: python -m build
   └─ Получаете: clothing_classifier-1.0.0-py3-none-any.whl

2. В Colab:
   ├─ Загружаете .whl файл
   ├─ Устанавливаете: pip install *.whl
   ├─ Загружаете данные
   └─ Запускаете обучение
```

---

## ✨ Что мы добавили:

1. ✅ **Vision Transformer** - современная архитектура через `timm`
2. ✅ **50 эпох** - больше времени на обучение
3. ✅ **pyproject.toml** - для сборки пакета
4. ✅ **Скрипт сборки** - `build_wheel.py`
5. ✅ **Инструкции** - для работы в Colab

---

## 🎓 Доступные модели:

- `vit_base_patch16_224` - Vision Transformer (по умолчанию)
- `vit_large_patch16_224` - Большой ViT
- `deit_base_distilled_patch16_224` - DeiT
- `efficientnet_b0` - EfficientNet
- `resnet50` - ResNet

Измените в `CONFIG['model_name']` для выбора модели.
