import pandas as pd 
import json 
database = pd.read_csv("rsna-breast-cancer-detection/train.csv")
print(database.query('difficult_negative_case == True'))
def createdatabase():
    with open('labels.json') as f:
        example = json.load(f)

    example["annotations"] = []
    example["images"] = []
    idimage = 1
    idannotations = 1
    for _,row in database.iterrows():

        dicti = {"id":idimage, "file_name": f"{row['patient_id']}@{row['image_id']}.png","width":1024,"height":2048, "license": 1,"coco_url": None}
        example["images"].append(dicti)
        dicti = {"id":idannotations,"image_id":idimage,"category_id": row['cancer'],"iscrowd":0, "bbox":[0,0,1024,2048],"area": 2097152,"segmentation": []}
        example["annotations"].append(dicti)
        idimage += 1
        idannotations += 1
    with open("testa.json", "w") as f:
        json.dump(example,f)