"""
Скраппер для Avito - собирает объявления с изображениями и текстом
"""
import os
import json
import time
from datetime import datetime
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


class AvitoScraper:
    def __init__(self, output_dir="avito_data", download_images=True):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.download_images = download_images
        
        if self.download_images:
            os.makedirs(self.images_dir, exist_ok=True)
        else:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Настройка Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Без графического интерфейса
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        print("[ИНИЦИАЛИЗАЦИЯ] Запуск браузера...")
        # Selenium 4.6+ автоматически управляет драйверами
        self.driver = webdriver.Chrome(options=chrome_options)
        print("[OK] Браузер запущен")
    
    def search_avito(self, query="ноутбук", limit=5, region="moskva", start_id=0):
        """
        Ищет объявления на Avito с поддержкой пагинации
        
        Args:
            query: Поисковый запрос
            limit: Количество объявлений для сбора
            region: Регион (moskva, sankt-peterburg, rossiya и т.д.)
            start_id: Начальный ID для нумерации (для продолжения сбора)
        """
        print(f"[ПОИСК] Ищу на Avito: '{query}' в регионе '{region}'")
        print(f"[ЦЕЛЬ] Собрать {limit} объявлений")
        
        items_data = []
        page = 1
        items_per_page = 50  # Примерно 50 объявлений на странице Avito
        
        try:
            while len(items_data) < limit:
                # URL страницы с пагинацией
                if page == 1:
                    search_url = f"https://www.avito.ru/{region}?q={query}"
                else:
                    search_url = f"https://www.avito.ru/{region}?q={query}&p={page}"
                
                print(f"\n[СТРАНИЦА {page}] Загрузка... (собрано: {len(items_data)}/{limit})")
                
                self.driver.get(search_url)
                time.sleep(3)  # Ждем загрузки
                
                # Получаем HTML
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                # Ищем карточки объявлений
                items = soup.find_all('div', {'data-marker': 'item'})
                
                if not items:
                    print("[ПРЕДУПРЕЖДЕНИЕ] Объявления не найдены. Пробую альтернативный селектор...")
                    items = soup.find_all('div', class_=lambda x: x and 'item-item' in x.lower())
                
                if not items:
                    print(f"[КОНЕЦ] Больше нет объявлений (страница {page})")
                    break
                
                print(f"[НАЙДЕНО] {len(items)} объявлений на странице {page}")
                
                # Обрабатываем объявления
                for item in items:
                    if len(items_data) >= limit:
                        break
                    
                    try:
                        # Используем start_id + текущее количество для глобальной нумерации
                        current_global_id = start_id + len(items_data)
                        item_data = self._parse_item(item, current_global_id)
                        
                        if item_data and item_data['url']:
                            # Парсим полную страницу для расширенных метаданных
                            extended_data = self._parse_full_page(item_data['url'])
                            if extended_data:
                                item_data.update(extended_data)
                            
                            items_data.append(item_data)
                            
                            # Выводим прогресс каждые 5 объявлений
                            if len(items_data) % 5 == 0:
                                print(f"[ПРОГРЕСС] {len(items_data)}/{limit} - {item_data['title'][:40]}...")
                            
                            # Сохраняем промежуточные результаты каждые 50 объявлений
                            if len(items_data) % 50 == 0:
                                self.save_metadata(items_data, f"metadata_checkpoint_{len(items_data)}.json")
                                print(f"[CHECKPOINT] Сохранен промежуточный результат: {len(items_data)} объявлений")
                    except Exception as e:
                        print(f"[ОШИБКА] Не удалось обработать объявление: {e}")
                    
                    time.sleep(0.5)  # Задержка между объявлениями
                
                page += 1
                
                # Задержка между страницами
                time.sleep(2)
                
                # Если на странице меньше объявлений, чем ожидалось - возможно это последняя страница
                if len(items) < 10:
                    print(f"[КОНЕЦ] Достигнут конец результатов поиска")
                    break
            
            return items_data
            
        except Exception as e:
            print(f"[ОШИБКА] Ошибка при скрапинге: {e}")
            # Сохраняем то, что успели собрать
            if items_data:
                self.save_metadata(items_data, "metadata_error_recovery.json")
                print(f"[ВОССТАНОВЛЕНИЕ] Сохранено {len(items_data)} объявлений до ошибки")
            return items_data
    
    def _parse_item(self, item, idx):
        """Парсит одно объявление из списка"""
        data = {
            'id': idx + 1,
            'title': '',
            'price': '',
            'description': '',
            'location': '',
            'seller_name': '',
            'category': [],
            'date_published': '',
            'views': '',
            'url': '',
            'image_url': '',
            'local_image_path': ''
        }
        
        # Заголовок
        title_elem = item.find('a', {'data-marker': 'item-title'})
        if not title_elem:
            title_elem = item.find('h3')
        if not title_elem:
            title_elem = item.find('a', {'itemprop': 'url'})
        if title_elem:
            data['title'] = title_elem.get_text(strip=True)
            # URL объявления
            href = title_elem.get('href', '')
            if href:
                data['url'] = f"https://www.avito.ru{href}" if href.startswith('/') else href
        
        # Цена - улучшенный парсинг
        price_elem = item.find('span', {'data-marker': 'item-price'})
        if not price_elem:
            price_elem = item.find('meta', {'itemprop': 'price'})
            if price_elem:
                data['price'] = price_elem.get('content', '')
            else:
                # Ищем по классу
                price_elem = item.find('span', class_=lambda x: x and 'price' in x.lower() if x else False)
                if price_elem:
                    data['price'] = price_elem.get_text(strip=True)
        else:
            data['price'] = price_elem.get_text(strip=True)
        
        # Описание - УЛУЧШЕННЫЙ ПАРСИНГ
        # Пробуем разные селекторы
        desc_elem = item.find('div', {'data-marker': 'item-description'})
        if not desc_elem:
            desc_elem = item.find('div', {'itemprop': 'description'})
        if not desc_elem:
            desc_elem = item.find('p', class_=lambda x: x and 'description' in x.lower() if x else False)
        if not desc_elem:
            # Ищем любой текстовый блок, который может быть описанием
            desc_elem = item.find('div', class_=lambda x: x and any(word in x.lower() for word in ['text', 'desc']) if x else False)
        
        if desc_elem:
            desc_text = desc_elem.get_text(strip=True)
            # Убираем очень короткие "описания" (вероятно не то)
            if len(desc_text) > 10:
                data['description'] = desc_text
        
        # Местоположение - УЛУЧШЕННЫЙ ПАРСИНГ
        # Пробуем разные варианты
        location_elem = item.find('div', {'data-marker': 'item-address'})
        if not location_elem:
            location_elem = item.find('span', {'data-marker': 'item-address'})
        if not location_elem:
            location_elem = item.find('div', class_=lambda x: x and 'geo' in x.lower() if x else False)
        if not location_elem:
            location_elem = item.find('span', class_=lambda x: x and any(word in x.lower() for word in ['address', 'location', 'geo']) if x else False)
        if not location_elem:
            # Ищем по itemprop
            location_elem = item.find('span', {'itemprop': 'address'})
        
        if location_elem:
            loc_text = location_elem.get_text(strip=True)
            if loc_text:
                data['location'] = loc_text
        
        # Изображение
        img_elem = item.find('img')
        if img_elem:
            img_url = img_elem.get('src', '')
            if not img_url:
                img_url = img_elem.get('data-src', '')
            if not img_url:
                # Пробуем srcset
                srcset = img_elem.get('srcset', '')
                if srcset:
                    # Берем первый URL из srcset
                    img_url = srcset.split(',')[0].split()[0]
            
            if img_url and not img_url.startswith('data:'):
                data['image_url'] = img_url
                
                # Скачиваем изображение только если включена опция
                if self.download_images:
                    # Скачиваем все изображения (их всего 400)
                    img_filename = f"avito_{idx + 1}.jpg"
                    img_path = os.path.join(self.images_dir, img_filename)
                    
                    if self.download_image(img_url, img_path):
                        data['local_image_path'] = img_filename
        
        return data if data['title'] else None
    
    def _parse_full_page(self, url):
        """Парсит полную страницу объявления для расширенных метаданных"""
        try:
            self.driver.get(url)
            time.sleep(2)  # Ждем загрузки
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            extended = {}
            
            # Полное описание
            desc = soup.find('div', {'data-marker': 'item-view/item-description'})
            if desc:
                desc_text = desc.get_text(strip=True)
                if len(desc_text) > 10:
                    extended['description'] = desc_text
            
            # Имя продавца
            seller = soup.find('div', {'data-marker': 'seller-info/name'})
            if not seller:
                seller = soup.find('span', {'itemprop': 'name'})
            if seller:
                extended['seller_name'] = seller.get_text(strip=True)
            
            # Категория (breadcrumbs)
            breadcrumbs = soup.find_all('span', {'itemprop': 'name'})
            if breadcrumbs:
                extended['category'] = [b.get_text(strip=True) for b in breadcrumbs]
            
            # Дата публикации
            date_elem = soup.find('span', {'data-marker': 'item-view/item-date'})
            if date_elem:
                extended['date_published'] = date_elem.get_text(strip=True)
            
            # Просмотры
            views = soup.find('span', {'data-marker': 'item-view/total-views'})
            if views:
                extended['views'] = views.get_text(strip=True)
            
            return extended
            
        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось получить расширенные метаданные: {e}")
            return {}
    
    def download_image(self, url, filepath):
        """Скачивает изображение"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"[ОШИБКА] Не удалось скачать изображение: {e}")
            return False
    
    def save_metadata(self, data, filename="metadata.json"):
        """Сохраняет данные в JSON"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[СОХРАНЕНО] Данные сохранены в {filename}")
    
    def run(self, query="ноутбук", limit=5, region="moskva"):
        """Запускает скрапинг"""
        start_time = datetime.now()
        
        print("="*60)
        print("AVITO SCRAPER - ЗАПУСК")
        print("="*60)
        print(f"[ВРЕМЯ СТАРТА] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[ПАРАМЕТРЫ] Запрос: '{query}', Лимит: {limit}, Регион: {region}")
        print(f"[СКАЧИВАНИЕ ИЗОБРАЖЕНИЙ] {'Включено' if self.download_images else 'Отключено'}")
        print("="*60)
        
        data = self.search_avito(query, limit, region)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if data:
            self.save_metadata(data)
            print("\n" + "="*60)
            print(f"[ГОТОВО] Собрано {len(data)} объявлений")
            print(f"[ВРЕМЯ РАБОТЫ] {duration}")
            print(f"[СКОРОСТЬ] {len(data) / duration.total_seconds():.2f} объявлений/сек")
            print(f"[ПАПКА] Данные сохранены в: {self.output_dir}")
            print("="*60)
            
            # Выводим краткую сводку (только первые 10)
            print("\n[СВОДКА] Первые 10 объявлений:")
            for item in data[:10]:
                print(f"  - {item['title'][:60]}")
                print(f"    Цена: {item['price']}")
        else:
            print("\n[ОШИБКА] Не удалось собрать данные")
        
        return data
    
    def close(self):
        """Закрывает браузер"""
        if self.driver:
            self.driver.quit()
            print("[ЗАВЕРШЕНИЕ] Браузер закрыт")


def main():
    scraper = None
    try:
        # Создаем скраппер
        # download_images=True - скачивать изображения
        scraper = AvitoScraper(
            output_dir="avito_data_fashion",  # Папка для модной одежды
            download_images=True  # Включаем скачивание изображений
        )
        
        # Модная одежда и обувь - 50 категорий по 100 фото = 5000
        # Цель: собрать по 100 фото на каждую категорию для обучения классификатора
        categories = [
            # Верхняя одежда
            {"query": "стильное пальто", "limit": 100},
            {"query": "модная косуха кожаная куртка", "limit": 100},
            {"query": "крутая джинсовка", "limit": 100},
            {"query": "трендовая парка", "limit": 100},
            {"query": "стильный пуховик", "limit": 100},
            {"query": "модная дубленка", "limit": 100},
            {"query": "крутая шуба", "limit": 100},
            {"query": "стильный плащ", "limit": 100},
            
            # Куртки и пиджаки
            {"query": "модный пиджак", "limit": 100},
            {"query": "стильный блейзер", "limit": 100},
            {"query": "крутая ветровка", "limit": 100},
            {"query": "бомбер куртка", "limit": 100},
            {"query": "стильная жилетка", "limit": 100},
            {"query": "модный кардиган", "limit": 100},
            
            # Брюки и джинсы
            {"query": "стильные брюки", "limit": 100},
            {"query": "модные джинсы", "limit": 100},
            {"query": "трендовые карго", "limit": 100},
            {"query": "классические брюки", "limit": 100},
            {"query": "стильные чиносы", "limit": 100},
            {"query": "модные шорты", "limit": 100},
            
            # Рубашки и топы
            {"query": "стильная рубашка", "limit": 100},
            {"query": "модная майка", "limit": 100},
            {"query": "крутой лонгслив", "limit": 100},
            {"query": "трендовая футболка", "limit": 100},
            {"query": "стильная блузка", "limit": 100},
            {"query": "модный свитшот", "limit": 100},
            {"query": "крутой худи", "limit": 100},
            {"query": "стильный толстовка", "limit": 100},
            {"query": "модная водолазка", "limit": 100},
            {"query": "трендовая кофта", "limit": 100},
            
            # Обувь
            {"query": "стильные кроссовки", "limit": 100},
            {"query": "модные кеды", "limit": 100},
            {"query": "крутые найк nike", "limit": 100},
            {"query": "трендовые ботинки", "limit": 100},
            {"query": "стильные сапоги", "limit": 100},
            {"query": "модные туфли", "limit": 100},
            {"query": "крутые кроссовки adidas", "limit": 100},
            {"query": "стильные лоферы", "limit": 100},
            {"query": "модные мокасины", "limit": 100},
            {"query": "трендовая обувь", "limit": 100},
            
            # Платья и юбки
            {"query": "модное платье", "limit": 100},
            {"query": "стильная юбка", "limit": 100},
            {"query": "крутое вечернее платье", "limit": 100},
            {"query": "трендовое летнее платье", "limit": 100},
            
            # Аксессуары
            {"query": "стильная сумка", "limit": 100},
            {"query": "модный рюкзак", "limit": 100},
            {"query": "крутые очки солнцезащитные", "limit": 100},
            {"query": "стильный ремень", "limit": 100},
            {"query": "модная шапка", "limit": 100},
            {"query": "трендовый шарф", "limit": 100},
            {"query": "стильные перчатки", "limit": 100},
            {"query": "модные часы", "limit": 100}
        ]
        
        # Проверяем сколько уже собрано
        try:
            metadata_file = os.path.join("avito_data_fashion", "metadata_final.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    start_count = len(existing_data)
                    print(f"[НАЙДЕНО] Уже собрано: {start_count} объявлений")
            else:
                start_count = 0
        except:
            start_count = 0
        
        all_data = []
        total_target = 5000  # 50 категорий x 100 фото = 5000 фото для обучения классификатора
        
        print("\n" + "="*60)
        print(f"СБОР {total_target} ФОТО ДЛЯ ОБУЧЕНИЯ КЛАССИФИКАТОРА")
        print("50 категорий одежды/обуви x 100 фото на категорию")
        print("С РАСШИРЕННЫМИ МЕТАДАННЫМИ")
        print(f"[НАЧАЛО] Продолжаем с {start_count} объявлений")
        print(f"[ЦЕЛЬ] Дособрать до 100 фото на каждую категорию")
        print("="*60)
        
        # Загружаем существующие данные если есть
        if start_count > 0:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
                print(f"[ЗАГРУЖЕНО] {len(all_data)} существующих объявлений")
                
                # Определяем сколько объявлений уже собрано по каждой категории
                # Ищем ключевые слова из запроса в title
                category_counts = {}
                for item in all_data:
                    title = item.get('title', '').lower()
                    # Проверяем каждую категорию
                    for cat in categories:
                        query = cat['query'].lower()
                        # Извлекаем ключевое слово (первое существительное)
                        keywords = [w for w in query.split() if len(w) > 3]  # Слова длиннее 3 символов
                        if keywords and any(kw in title for kw in keywords):
                            category_counts[cat['query']] = category_counts.get(cat['query'], 0) + 1
                            break
                
                # Выводим статистику по первым 10 категориям
                print(f"[АНАЛИЗ] Уже собрано по категориям (первые 10):")
                for cat_query, count in list(category_counts.items())[:10]:
                    needed = categories[[c['query'] for c in categories].index(cat_query)]['limit']
                    print(f"  {cat_query}: {count}/{needed}")
            except Exception as e:
                print(f"[ОШИБКА] Не удалось загрузить существующие данные: {e}")
                all_data = []
        else:
            all_data = []
        
        for idx, category in enumerate(categories):
            if len(all_data) >= total_target:
                break
            
            remaining = total_target - len(all_data)
            if remaining <= 0:
                break
            
            # Проверяем, не собрана ли уже эта категория полностью
            already_collected = category_counts.get(category['query'], 0) if start_count > 0 else 0
            if already_collected >= category['limit']:
                print(f"\n[ПРОПУСК] Категория '{category['query']}' уже собрана ({already_collected}/{category['limit']})")
                continue
            
            print(f"\n[КАТЕГОРИЯ {idx+1}/{len(categories)}] {category['query']}")
            if already_collected > 0:
                print(f"[ПРОДОЛЖЕНИЕ] Уже собрано: {already_collected}, нужно еще: {category['limit'] - already_collected}")
            print("-"*60)
            
            # Собираем данные по категории (учитываем уже собранные)
            category_limit = min(category['limit'] - already_collected, remaining)
            if category_limit <= 0:
                continue
                
            data = scraper.search_avito(
                query=category['query'],
                limit=category_limit,
                region="moskva",
                start_id=len(all_data)  # Передаем текущий счетчик для нумерации
            )
            
            all_data.extend(data)
            
            print(f"[ИТОГО СОБРАНО] {len(all_data)}/{total_target}")
            
            # Сохраняем промежуточный результат каждые 500 объявлений
            if len(all_data) % 500 == 0 or len(all_data) >= total_target:
                scraper.save_metadata(all_data, f"metadata_diverse_{len(all_data)}.json")
        
        # Финальное сохранение
        scraper.save_metadata(all_data, "metadata_final.json")
        
        print("\n" + "="*60)
        print(f"[ФИНАЛ] Собрано {len(all_data)} разнообразных объявлений")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n[ПРЕРВАНО] Остановлено пользователем")
    except Exception as e:
        print(f"\n[ОШИБКА] Произошла ошибка: {e}")
    finally:
        if scraper:
            scraper.close()


if __name__ == "__main__":
    main()
