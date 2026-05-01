"""
Обучение классификатора изображений одежды
Использует Transfer Learning с Vision Transformer или EfficientNet-B0
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
from collections import defaultdict
import time
from datetime import datetime
# Для Vision Transformer
import timm

CONFIG = {
    'data_dir': Path("classifier_data"),
    'model_name': 'vit_base_patch16_224',  # Vision Transformer или 'efficientnet_b0', 'resnet50'
    'num_epochs': 50,
    'batch_size': 64,
    'learning_rate_classifier': 0.001,
    'learning_rate_unfrozen': 0.0001,
    'weight_decay': 0.005,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'save_dir': Path("models"),
    'use_amp': True,  # Mixed precision — ускорение на GPU
}

print(f"[УСТРОЙСТВО] Используется: {CONFIG['device']}")

class ClothingDataset(Dataset):
    """Датасет для изображений одежды"""
    
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Загружаем список категорий (БЕЗ маппинга - используем категории напрямую)
        mapping_file = Path(data_dir) / "category_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                # Используем categories напрямую (имена папок = категории)
                self.categories = sorted(mapping_data.get('categories', []))
        else:
            # Если файла нет, собираем категории из папок напрямую
            self.categories = []
            if self.data_dir.exists():
                for cat_dir in self.data_dir.iterdir():
                    if cat_dir.is_dir():
                        self.categories.append(cat_dir.name)
            self.categories = sorted(self.categories)
        
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Собираем все изображения
        for category in self.categories:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                continue
            
            label = self.category_to_idx[category]
            for img_path in category_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(label)
        
        print(f"[{split.upper()}] Загружено {len(self.images)} изображений, {len(self.categories)} классов")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"[ОШИБКА] Не удалось загрузить {img_path}: {e}")
            # Возвращаем черное изображение как fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224)))
            return image, label

def get_transforms():
    """Возвращает трансформации для train и val"""
    
    # аугментация для обучения
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Простые трансформации для валидации
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes, model_name='vit_base_patch16_224'):
    """Создает модель с поддержкой Vision Transformer и EfficientNet"""
    
    if model_name.startswith('vit_'):
        # Vision Transformer через timm
        print(f"[ЗАГРУЗКА] Vision Transformer: {model_name}...")
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
        print("[OK] Vision Transformer загружен")
        
        # Замораживаем все параметры
        for param in model.parameters():
            param.requires_grad = False
        
        # Размораживаем последние блоки transformer — больше блоков = быстрее рост точности
        if hasattr(model, 'blocks'):
            num_blocks = len(model.blocks)
            for i in range(max(0, num_blocks - 3), num_blocks):  # последние 3 блока
                for param in model.blocks[i].parameters():
                    param.requires_grad = True
        
        # Head всегда обучается
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        
        return model
    
    elif model_name == 'efficientnet_b0':
        from torchvision.models import EfficientNet_B0_Weights
        
        print("[ЗАГРУЗКА] Пытаемся загрузить предобученные веса EfficientNet-B0...")
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        print("[OK] Предобученные веса загружены")
        
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.features[6].parameters():
            param.requires_grad = True
        for param in model.features[7].parameters():
            param.requires_grad = True
        
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Увеличено для регуляризации (было 0.2)
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
        return model
    
    elif model_name == 'resnet50':
        from torchvision.models import ResNet50_Weights
        
        print("[ЗАГРУЗКА] Загружаю ResNet50...")
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        print("[OK] ResNet50 загружен")
        
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
        
        return model
    
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    # device может быть строкой ('cuda' / 'cpu') или torch.device
    is_cuda = getattr(device, "type", None) == "cuda" or str(device) == "cuda"
    non_blocking = is_cuda

    for images, labels in dataloader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        if use_amp and is_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16): # bfloat16 лучше TODO:
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # device может быть строкой ('cuda' / 'cpu') или torch.device
    is_cuda = getattr(device, "type", None) == "cuda" or str(device) == "cuda"
    non_blocking = is_cuda

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if use_amp and is_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def main():
    print("="*60)
    print("ОБУЧЕНИЕ КЛАССИФИКАТОРА ИЗОБРАЖЕНИЙ ОДЕЖДЫ")
    print("="*60)
    print(f"Модель: {CONFIG['model_name']}")
    print(f"Эпох: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print("="*60)

    train_transform, val_transform = get_transforms()
    
    train_dataset = ClothingDataset(CONFIG['data_dir'], 'train', train_transform)
    val_dataset = ClothingDataset(CONFIG['data_dir'], 'val', val_transform)
    
    is_cuda = CONFIG['device'] == 'cuda'
    if is_cuda:
        torch.backends.cudnn.benchmark = True  # Ускорение свёрток на GPU
    num_workers = CONFIG['num_workers'] if is_cuda else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    num_classes = len(train_dataset.categories)
    print(f"\n[КЛАССЫ] Количество классов: {num_classes}")
    print(f"[КЛАССЫ] Первые 10: {', '.join(train_dataset.categories[:10])}...")

    model = create_model(num_classes, CONFIG['model_name'])
    model = model.to(CONFIG['device'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[МОДЕЛЬ] Всего параметров: {total_params:,}")
    print(f"[МОДЕЛЬ] Обучаемых параметров: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    if CONFIG['model_name'].startswith('vit_'):
        # Vision Transformer
        head_params = list(model.head.parameters()) if hasattr(model, 'head') else []
        unfrozen_params = []
        if hasattr(model, 'blocks'):
            num_blocks = len(model.blocks)
            for i in range(max(0, num_blocks - 3), num_blocks):
                unfrozen_params.extend(list(model.blocks[i].parameters()))
    elif CONFIG['model_name'] == 'efficientnet_b0':
        # EfficientNet
        head_params = list(model.classifier.parameters())
        unfrozen_params = list(model.features[6].parameters()) + list(model.features[7].parameters())
    elif CONFIG['model_name'] == 'resnet50':
        # ResNet
        head_params = list(model.fc.parameters())
        unfrozen_params = list(model.layer3.parameters()) + list(model.layer4.parameters())

    # Создаем оптимизатор с weight decay для регуляризации
    weight_decay = CONFIG.get('weight_decay', 0.01)
    if unfrozen_params:
        optimizer = optim.Adam([
            {'params': head_params, 'lr': CONFIG['learning_rate_classifier'], 'weight_decay': weight_decay},
            {'params': unfrozen_params, 'lr': CONFIG['learning_rate_unfrozen'], 'weight_decay': weight_decay}
        ])
    else:
        optimizer = optim.Adam(head_params, lr=CONFIG['learning_rate_classifier'], weight_decay=weight_decay)

    # OneCycleLR — в начале высокий LR, точность растёт быстрее; к концу LR падает для стабилизации
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CONFIG['num_epochs']
    max_lr = (
        [CONFIG['learning_rate_classifier'], CONFIG['learning_rate_unfrozen']]
        if unfrozen_params else CONFIG['learning_rate_classifier']
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.15,   # 15% эпох — разгон LR
        div_factor=10.0,   # начальный LR = max_lr / 10
        final_div_factor=10.0,
    )

    CONFIG['save_dir'].mkdir(exist_ok=True)
    best_val_acc = 0.0

    use_amp = CONFIG.get('use_amp', False) and is_cuda
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("\n[AMP] Mixed precision включён (float16)")

    print("\n[ОБУЧЕНИЕ] Начало обучения...")
    print("-"*60)

    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device'],
            scaler=scaler, scheduler=scheduler
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG['device'], use_amp=use_amp
        )

        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = CONFIG['save_dir'] / f"best_model_{CONFIG['model_name']}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'categories': train_dataset.categories,
                'model_name': CONFIG['model_name']
            }, model_path)
            print(f"  [СОХРАНЕНО] Лучшая модель: {val_acc:.2f}%")

        print()

    print("\n" + "="*60)
    print("[ТЕСТИРОВАНИЕ] Загрузка лучшей модели для финальной оценки...")
    print("="*60)
    model_path = CONFIG['save_dir'] / f"best_model_{CONFIG['model_name']}.pth"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[ЗАГРУЖЕНО] Модель из эпохи {checkpoint['epoch']+1}")
        print(f"[ЗАГРУЖЕНО] Val accuracy: {checkpoint['val_acc']:.2f}%")

    print("\n[ТЕСТИРОВАНИЕ] Оценка на test выборке...")
    test_dataset = ClothingDataset(CONFIG['data_dir'], 'test', val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    test_loss, test_acc = validate(
        model, test_loader, criterion, CONFIG['device'],
        use_amp=CONFIG.get('use_amp', False) and CONFIG['device'] == 'cuda'
    )
    
    print("="*60)
    print(f"[ЗАВЕРШЕНО] Лучшая точность на валидации: {best_val_acc:.2f}%")
    print(f"[ТЕСТ] Финальная точность на test: {test_acc:.2f}%")
    print(f"[ТЕСТ] Test loss: {test_loss:.4f}")
    print(f"[СОХРАНЕНО] Модель: {model_path}")
    print("="*60)

if __name__ == "__main__":
    main()
