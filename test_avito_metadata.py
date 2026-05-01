"""
Тестовый скрипт для проверки ВСЕХ доступных метаданных на Avito
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import json
import time

# Настройка браузера
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

print("[ТЕСТ] Проверка доступных метаданных на Avito...")
driver = webdriver.Chrome(options=chrome_options)

try:
    # Берем одно объявление для детального анализа
    test_url = "https://www.avito.ru/moskva/noutbuki/igrovye_noutbuki_dlya_raboty_i_ucheby_s_garantiey_2493039373"
    
    print(f"[ЗАГРУЗКА] {test_url}")
    driver.get(test_url)
    time.sleep(5)  # Даем время загрузиться
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Сохраняем HTML для анализа
    with open('avito_page_sample.html', 'w', encoding='utf-8') as f:
        f.write(soup.prettify())
    print("[СОХРАНЕНО] avito_page_sample.html")
    
    # Ищем все возможные метаданные
    metadata = {}
    
    # 1. Заголовок
    title = soup.find('h1', {'data-marker': 'item-view/title-info'})
    if title:
        metadata['title'] = title.get_text(strip=True)
    
    # 2. Цена
    price = soup.find('span', {'data-marker': 'item-view/item-price'})
    if price:
        metadata['price'] = price.get_text(strip=True)
    
    # 3. Описание
    desc = soup.find('div', {'data-marker': 'item-view/item-description'})
    if desc:
        metadata['description'] = desc.get_text(strip=True)
    
    # 4. Адрес/Местоположение
    address = soup.find('span', {'data-marker': 'delivery/location'})
    if not address:
        address = soup.find('div', class_=lambda x: x and 'geo' in x.lower() if x else False)
    if address:
        metadata['address'] = address.get_text(strip=True)
    
    # 5. Дата публикации
    date_elem = soup.find('span', {'data-marker': 'item-view/item-date'})
    if date_elem:
        metadata['date_published'] = date_elem.get_text(strip=True)
    
    # 6. Просмотры
    views = soup.find('span', {'data-marker': 'item-view/total-views'})
    if views:
        metadata['views'] = views.get_text(strip=True)
    
    # 7. Имя продавца
    seller = soup.find('div', {'data-marker': 'seller-info/name'})
    if seller:
        metadata['seller_name'] = seller.get_text(strip=True)
    
    # 8. Рейтинг продавца
    rating = soup.find('div', {'data-marker': 'seller-info/rating'})
    if rating:
        metadata['seller_rating'] = rating.get_text(strip=True)
    
    # 9. Категория
    breadcrumbs = soup.find_all('span', {'itemprop': 'name'})
    if breadcrumbs:
        metadata['category'] = [b.get_text(strip=True) for b in breadcrumbs]
    
    # 10. Характеристики товара
    params = soup.find_all('li', {'data-marker': 'item-view/item-params'})
    if params:
        metadata['characteristics'] = []
        for param in params:
            metadata['characteristics'].append(param.get_text(strip=True))
    
    # 11. Доставка
    delivery = soup.find('div', {'data-marker': 'delivery/delivery-info'})
    if delivery:
        metadata['delivery_available'] = delivery.get_text(strip=True)
    
    # 12. Изображения (все)
    images = soup.find_all('div', {'data-marker': 'image-frame/image-wrapper'})
    if images:
        metadata['images_count'] = len(images)
    
    # 13. ID объявления
    item_id = soup.find('span', {'data-marker': 'item-view/item-id'})
    if item_id:
        metadata['item_id'] = item_id.get_text(strip=True)
    
    # Сохраняем результат
    with open('avito_metadata_test.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("[РЕЗУЛЬТАТ] Найденные метаданные:")
    print("="*60)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    print("\n[СОХРАНЕНО] avito_metadata_test.json")
    
    # Ищем ВСЕ data-marker атрибуты для дальнейшего анализа
    all_markers = soup.find_all(attrs={'data-marker': True})
    markers_list = list(set([elem.get('data-marker') for elem in all_markers]))
    
    with open('avito_all_markers.txt', 'w', encoding='utf-8') as f:
        f.write("ВСЕ НАЙДЕННЫЕ DATA-MARKERS:\n")
        f.write("="*60 + "\n")
        for marker in sorted(markers_list):
            f.write(f"{marker}\n")
    
    print(f"[НАЙДЕНО] {len(markers_list)} уникальных data-marker атрибутов")
    print("[СОХРАНЕНО] avito_all_markers.txt")

finally:
    driver.quit()
    print("\n[ЗАВЕРШЕНО] Тест метаданных завершен")
