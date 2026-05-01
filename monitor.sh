#!/bin/bash
# Скрипт для мониторинга прогресса скраппера

echo "=== МОНИТОРИНГ AVITO SCRAPER ==="
echo ""

# Проверяем процесс
if ps aux | grep -v grep | grep "python avito_scraper.py" > /dev/null; then
    echo "[СТАТУС] Скраппер работает ✓"
else
    echo "[СТАТУС] Скраппер не запущен ✗"
fi

echo ""

# Количество скачанных изображений
if [ -d "avito_data/images" ]; then
    img_count=$(ls -1 avito_data/images/*.jpg 2>/dev/null | wc -l)
    echo "[ИЗОБРАЖЕНИЯ] Скачано: $img_count"
else
    echo "[ИЗОБРАЖЕНИЯ] Папка не создана"
fi

echo ""

# Размер данных
if [ -f "avito_data/metadata_diverse_1000.json" ]; then
    size=$(du -h avito_data/metadata_diverse_1000.json | cut -f1)
    echo "[CHECKPOINT] metadata_diverse_1000.json ($size)"
fi

if [ -f "avito_data/metadata_final.json" ]; then
    size=$(du -h avito_data/metadata_final.json | cut -f1)
    lines=$(wc -l < avito_data/metadata_final.json)
    echo "[ФИНАЛ] metadata_final.json ($size, $lines строк)"
fi

echo ""

# Последние 20 строк лога
if [ -f "scraper_log.txt" ]; then
    echo "[ЛОГ] Последние 20 строк:"
    echo "---"
    tail -n 20 scraper_log.txt
else
    echo "[ЛОГ] Файл не найден"
fi
