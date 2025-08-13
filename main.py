from ultralytics import YOLO

def main():
    model = YOLO('yolov8s-seg.pt')

    model.train(

        data='/home/user/ai_hackathon/dataset/data.yaml',
        epochs=300,               
        batch=16,                 
        imgsz=768,
        workers=4,                 
        device=0,                  
        patience=300,               
        optimizer="AdamW",           
        lr0=0.0001,                 
        weight_decay=0.0005,       
        augment=True
    )

    # results = model.val()
    # print(results)


if __name__ == '__main__':
    main()
