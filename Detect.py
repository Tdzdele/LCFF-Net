from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/LCFF-Net/LCFF-Net-t/weights/best.pt')
    model.predict(source='datasets/VisDrone/VisDrone2019-DET-val/images',
                  imgsz=640,
                  project='runs/detect/val',
                  name='LCFF-Net-t',
                  save=True,
                  show_labels=False,
                )