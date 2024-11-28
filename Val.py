from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/train/LCFF-Net/LCFF-Net-t/weights/best.pt')
    model.val(data='VisDrone.yaml',
              split='val',
              imgsz=640,
              batch=16,
              device='0, 1',
              save_json=True,
              plots=True,
              project='runs/val',
              name='LCFF-Net-t',
              )