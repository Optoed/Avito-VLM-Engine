"""
Организация данных для обучения классификатора
Работает с 50 категориями БЕЗ объединения
"""
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

# 50 категорий из скраппера (БЕЗ объединения, БЕЗ прилагательных)
CATEGORIES = [
    # Верхняя одежда
    "пальто",
    "косуха кожаная куртка",
    "джинсовка",
    "парка",
    "пуховик",
    "дубленка",
    "шуба",
    "плащ",
    
    # Куртки и пиджаки
    "пиджак",
    "блейзер",
    "ветровка",
    "бомбер куртка",
    "жилетка",
    "кардиган",
    
    # Брюки и джинсы
    "брюки",
    "джинсы",
    "карго",
    "классические брюки",
    "чиносы",
    "шорты",
    
    # Рубашки и топы
    "рубашка",
    "майка",
    "лонгслив",
    "футболка",
    "блузка",
    "свитшот",
    "худи",
    "толстовка",
    "водолазка",
    "кофта",
    
    # Обувь
    "кроссовки",
    "кеды",
    "найк nike",
    "ботинки",
    "сапоги",
    "туфли",
    "кроссовки adidas",
    "лоферы",
    "мокасины",
    "обувь",
    
    # Платья и юбки
    "платье",
    "юбка",
    "вечернее платье",
    "летнее платье",
    
    # Аксессуары
    "сумка",
    "рюкзак",
    "очки солнцезащитные",
    "ремень",
    "шапка",
    "шарф",
    "перчатки",
    "часы"
]

def normalize_category_name(category_query):
    """Использует запрос напрямую как имя папки (без маппинга)"""
    # Заменяем пробелы на подчеркивания для имен папок
    return category_query.lower().replace(' ', '_')

def categorize_item(item):
    """Определяет категорию товара по названию"""
    title = item.get('title', '').lower()
    category_list = item.get('category', [])
    
    # Ищем ключевые слова из каждой категории в названии
    for category_query in CATEGORIES:
        query_lower = category_query.lower()
        # Извлекаем ключевые слова (слова длиннее 3 символов)
        keywords = [w for w in query_lower.split() if len(w) > 3]
        
        # Проверяем совпадение ключевых слов
        if keywords and any(kw in title for kw in keywords):
            return category_query
    
    # Если не нашли в названии, проверяем категории из метаданных
    for cat in category_list:
        cat_lower = cat.lower()
        for category_query in CATEGORIES:
            query_lower = category_query.lower()
            keywords = [w for w in query_lower.split() if len(w) > 3]
            if keywords and any(kw in cat_lower for kw in keywords):
                return category_query
    
    # Если не нашли - категория "другое"
    return 'другое'

def organize_data():
    """Организует данные в структуру папок для обучения"""
    
    # Пути
    metadata_file = Path("avito_data_fashion/metadata_final.json")
    images_dir = Path("avito_data_fashion/images")
    output_dir = Path("classifier_data")
    
    # Создаем только базовые папки train/val/test
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    print("[ЗАГРУЗКА] Читаю метаданные...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[НАЙДЕНО] {len(data)} объявлений")
    
    # Группируем по категориям (только с фото!)
    categorized = defaultdict(list)
    
    for item in data:
        if not item.get('local_image_path'):
            continue  # Пропускаем без изображений
        
        image_path = images_dir / item['local_image_path']
        if not image_path.exists():
            continue  # Пропускаем если файл не существует
        
        category = categorize_item(item)
        categorized[category].append({
            'item': item,
            'image_path': image_path
        })
    
    print(f"\n[КАТЕГОРИИ] Распределение по категориям:")
    for cat, items in sorted(categorized.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(items)} изображений")
    
    # Разделяем на train/val/test (80/10/10)
    random.seed(42)  # Для воспроизводимости
    
    stats = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}
    
    for category, items in categorized.items():
        random.shuffle(items)
        
        n = len(items)
        if n <= 90:  # Пропускаем категории с 90 или менее фотографиями (оставляем только > 90)
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Категория '{category}' имеет только {n} фото (нужно > 90), пропускаем")
            continue
        
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]
        
        category_folder = normalize_category_name(category)
        
        # Копируем файлы
        for split, split_items in [('train', train_items), ('val', val_items), ('test', test_items)]:
            # Создаём папку категории только если есть файлы для копирования
            if split_items:
                category_path = output_dir / split / category_folder
                category_path.mkdir(parents=True, exist_ok=True)
            
            for item_data in split_items:
                src = item_data['image_path']
                dst = output_dir / split / category_folder / src.name
                
                try:
                    shutil.copy2(src, dst)
                    stats[split][category] += 1
                except Exception as e:
                    print(f"[ОШИБКА] Не удалось скопировать {src.name}: {e}")
    
    # Выводим статистику
    print("\n" + "="*60)
    print("[СТАТИСТИКА] Распределение данных:")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        total = sum(stats[split].values())
        print(f"\n[{split.upper()}] Всего: {total}")
        for cat in sorted(stats[split].keys(), key=lambda x: -stats[split][x]):
            count = stats[split][cat]
            print(f"  {cat}: {count}")
    
    # Сохраняем список категорий (БЕЗ маппинга - используем категории напрямую)
    mapping_file = output_dir / "category_mapping.json"
    # Собираем уникальные категории из папок (имена папок = категории)
    actual_categories = []
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            for cat_dir in split_dir.iterdir():
                if cat_dir.is_dir() and cat_dir.name not in actual_categories:
                    actual_categories.append(cat_dir.name)
    
    actual_categories = sorted(actual_categories)
    
    category_mapping = {
        'categories': actual_categories  # Просто список категорий без маппинга
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(category_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\n[СОХРАНЕНО] Список категорий: {mapping_file}")
    print(f"[КАТЕГОРИЙ] Найдено: {len(actual_categories)}")
    
    # Удаляем пустые папки категорий (без изображений)
    print("\n[ОЧИСТКА] Удаление пустых папок...")
    removed_count = 0
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            for cat_dir in list(split_dir.iterdir()):  # list() чтобы избежать изменения во время итерации
                if cat_dir.is_dir():
                    # Проверяем, есть ли изображения в папке
                    image_files = list(cat_dir.glob('*.jpg'))
                    if len(image_files) == 0:
                        try:
                            cat_dir.rmdir()
                            removed_count += 1
                            print(f"  Удалена пустая папка: {split}/{cat_dir.name}")
                        except Exception as e:
                            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось удалить {cat_dir.name}: {e}")
    
    if removed_count > 0:
        print(f"[ОЧИЩЕНО] Удалено пустых папок: {removed_count}")
    else:
        print("[ОЧИЩЕНО] Пустых папок не найдено")
    
    # Обновляем список категорий после удаления пустых папок
    actual_categories = []
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            for cat_dir in split_dir.iterdir():
                if cat_dir.is_dir():
                    image_files = list(cat_dir.glob('*.jpg'))
                    if len(image_files) > 0 and cat_dir.name not in actual_categories:
                        actual_categories.append(cat_dir.name)
    
    actual_categories = sorted(actual_categories)
    
    # Обновляем category_mapping.json с актуальным списком (только категории с данными)
    category_mapping = {
        'categories': actual_categories
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(category_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"[ОБНОВЛЕНО] Список категорий: {len(actual_categories)} категорий с данными")
    print(f"[ГОТОВО] Данные организованы в: {output_dir}")

if __name__ == "__main__":
    organize_data()
