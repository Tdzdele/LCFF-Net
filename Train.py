from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/LCFF-Net/LCFF-Net.yaml')
    model.train(data='VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=128,
                device='0, 1',
                optimizer='SGD',
                # patience=0, # close earlystop
                amp=True,
                project='runs/train/LCFF-Net',
                name='LCFF-Net',
                )