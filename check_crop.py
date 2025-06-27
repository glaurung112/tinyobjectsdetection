import os

print("Verificando que todas las imágenes tengan sus labels...")
print()

src = 'dataset/cropped_base'
imgs = [f for f in os.listdir(src) if f.endswith('.png') or f.endswith('.jpg')]
txts = [f for f in os.listdir(src) if f.endswith('.txt')]

print(f"Imágenes encontradas: {len(imgs)}")
print(f"Labels encontrados: {len(txts)}")

missing = [img for img in imgs if img.rsplit('.', 1)[0] + '.txt' not in txts]

if missing:
    print("Algunas imágenes no tienen label:")
    for m in missing:
        print(f" - {m}")
else:
    print("Todas las imágenes tienen su label.")
