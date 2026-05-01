"""
Скрипт для сборки wheel пакета
Использование: python build_wheel.py
"""
import subprocess
import sys
from pathlib import Path

def build_wheel():
    """Собирает wheel пакет"""
    print("="*60)
    print("СБОРКА WHEEL ПАКЕТА")
    print("="*60)
    
    # Проверяем наличие build
    try:
        import build
    except ImportError:
        print("[УСТАНОВКА] Устанавливаю build и wheel...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "build", "wheel"])
    
    # Собираем wheel
    print("\n[СБОРКА] Запускаю сборку...")
    subprocess.check_call([sys.executable, "-m", "build"])
    
    # Проверяем результат
    dist_dir = Path("dist")
    if dist_dir.exists():
        wheels = list(dist_dir.glob("*.whl"))
        if wheels:
            print(f"\n[УСПЕХ] Wheel создан: {wheels[0].name}")
            print(f"[РАЗМЕР] {wheels[0].stat().st_size / 1024 / 1024:.2f} MB")
            print(f"\n[ИНСТРУКЦИЯ] Загрузите файл в Google Colab:")
            print(f"  {wheels[0].absolute()}")
        else:
            print("[ОШИБКА] Wheel файл не найден")
    else:
        print("[ОШИБКА] Папка dist не создана")

if __name__ == "__main__":
    build_wheel()
