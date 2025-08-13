from ultralytics import YOLO

def main():
    model = YOLO('yolov8s-seg.pt')

    model.train(

        data='/home/user/ai_hackathon/dataset/data.yaml',
        epochs=300,               
        batch=16,                 
        imgsz=768,                 # 이미지 크기 확대 (메모리 허용 시)
        workers=4,                 
        device=0,                  
        patience=300,               
        optimizer="AdamW",           
        lr0=0.000000000000000001,                 # 기본 학습률 증가
        # lr_scheduler='cosine',     # cosine learning rate scheduler 추가
        weight_decay=0.0005,       
        augment=True,              # 내장 증강 유지
        # ema=True
    )

    # results = model.val()
    # print(results)


if __name__ == '__main__':
    main()