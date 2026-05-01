# Исправление переобучения (Overfitting)

## 🔴 Проблема: Переобучение

**Признаки:**
- ✅ Train accuracy растет: 44% → 75%
- ❌ Val accuracy падает: 53% → 42%
- ❌ Val loss растет: 1.59 → 2.11
- ✅ Train loss падает: 2.09 → 0.62

**Вывод:** Модель запоминает обучающие данные, но не обобщается.

---

## ✅ Исправления (уже внесены в код):

1. **Уменьшен Learning Rate:**
   - Head: 0.001 → 0.0001
   - Unfrozen layers: 0.0001 → 0.00001

2. **Добавлен Weight Decay (0.01):**
   - Регуляризация для предотвращения переобучения

3. **Увеличен Dropout:**
   - 0.2 → 0.3

4. **Уменьшен Batch Size:**
   - 32 → 16 (для стабильности)

5. **Уменьшен num_workers:**
   - 4 → 2 (для Colab)

6. **Уменьшен Early Stopping Patience:**
   - 7 → 5 (раньше остановится)

---

## 🚀 Что делать в Colab:

### Вариант 1: Пересобрать wheel (рекомендуется)

```python
# В Colab: Загрузите обновленный train_classifier.py
# Или пересоберите wheel на локальной машине и загрузите заново
```

### Вариант 2: Изменить параметры прямо в Colab

```python
# Добавьте эту ячейку ПЕРЕД запуском main()

import sys
if 'train_classifier' in sys.modules:
    del sys.modules['train_classifier']

from train_classifier import CONFIG

# Изменяем параметры
CONFIG['learning_rate_classifier'] = 0.0001  # Было 0.001
CONFIG['learning_rate_unfrozen'] = 0.00001   # Было 0.0001
CONFIG['batch_size'] = 16                     # Было 32
CONFIG['num_workers'] = 2                     # Было 4
CONFIG['early_stopping_patience'] = 5         # Было 7

# Добавляем weight_decay в оптимизатор
# (нужно изменить код оптимизатора)

from train_classifier import main
main()
```

### Вариант 3: Быстрое исправление в коде

```python
# Ячейка: Исправление переобучения
import torch
import torch.nn as nn
import torch.optim as optim

# Переопределяем CONFIG
from train_classifier import CONFIG, create_model, ClothingDataset, get_transforms
from torch.utils.data import DataLoader

CONFIG['learning_rate_classifier'] = 0.0001
CONFIG['learning_rate_unfrozen'] = 0.00001
CONFIG['batch_size'] = 16
CONFIG['num_workers'] = 2
CONFIG['early_stopping_patience'] = 5
CONFIG['weight_decay'] = 0.01

# Загружаем данные
train_transform, val_transform = get_transforms()
train_dataset = ClothingDataset(CONFIG['data_dir'], 'train', train_transform)
val_dataset = ClothingDataset(CONFIG['data_dir'], 'val', val_transform)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=True, num_workers=CONFIG['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                        shuffle=False, num_workers=CONFIG['num_workers'])

# Создаем модель
num_classes = len(train_dataset.categories)
model = create_model(num_classes, CONFIG['model_name'])
model = model.to(CONFIG['device'])

# Оптимизатор с weight_decay
criterion = nn.CrossEntropyLoss()

# Параметры для ViT
head_params = list(model.head.parameters()) if hasattr(model, 'head') else []
unfrozen_params = []
if hasattr(model, 'blocks'):
    num_blocks = len(model.blocks)
    for i in range(max(0, num_blocks - 2), num_blocks):
        unfrozen_params.extend(list(model.blocks[i].parameters()))

optimizer = optim.Adam([
    {'params': head_params, 'lr': CONFIG['learning_rate_classifier'], 'weight_decay': 0.01},
    {'params': unfrozen_params, 'lr': CONFIG['learning_rate_unfrozen'], 'weight_decay': 0.01}
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Импортируем функции обучения
from train_classifier import train_epoch, validate

# Обучение
CONFIG['save_dir'].mkdir(exist_ok=True)
best_val_acc = 0.0
patience_counter = 0

print("\n[ОБУЧЕНИЕ] Начало обучения с исправленными параметрами...")
print("-"*60)

for epoch in range(CONFIG['num_epochs']):
    import time
    start_time = time.time()
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
    val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
    
    scheduler.step(val_loss)
    epoch_time = time.time() - start_time
    
    print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        model_path = CONFIG['save_dir'] / f"best_model_{CONFIG['model_name']}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, model_path)
        print(f"  [СОХРАНЕНО] Лучшая модель: {val_acc:.2f}%")
    else:
        patience_counter += 1
    
    if patience_counter >= CONFIG['early_stopping_patience']:
        print(f"\n[ОСТАНОВКА] Early stopping после {epoch+1} эпох")
        break
    
    print()
```

---

## 📊 Ожидаемые результаты после исправления:

- ✅ Train и Val accuracy должны сближаться
- ✅ Val loss должен уменьшаться или стабилизироваться
- ✅ Разница между train и val не должна превышать 10-15%

---

## 💡 Дополнительные советы:

1. **Если переобучение продолжается:**
   - Уменьшите learning rate еще больше (0.00005)
   - Увеличьте dropout до 0.4
   - Заморозьте больше слоев (только последний блок)

2. **Если обучение слишком медленное:**
   - Увеличьте batch_size до 24
   - Увеличьте learning_rate_classifier до 0.0002

3. **Мониторинг:**
   - Следите за разницей между train и val accuracy
   - Если разница > 20% - это переобучение
