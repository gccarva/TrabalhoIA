import pandas as pd 
import json 
from sklearn.model_selection import train_test_split
import shutil
dirtrainimage= "train_images"
dirtestimage = "test_images"
dirvalidateimage = "val_images"
caminhoimages= "/home/fernando/Documents/ia/rsna-breast-cancer-detection/processed_images"
with open("labels.json") as f:
    labels = json.load(f)
df = pd.read_csv("/home/fernando/Documents/ia/rsna-breast-cancer-detection/train.csv")

# Primeiro, separe o conjunto de treino (70%) e o conjunto temporário (30% que será dividido em validate e test)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=30)

# Em seguida, separe o conjunto temporário em validate (10% do total) e test (20% do total)
validate_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=30)
labesttrain = dict()
labelsval = dict()
labelstest = dict()
labesttrain["info"] = labels["info"]
labesttrain["licenses"] = labels["licenses"]
labesttrain["categories"] = labels["categories"]
labesttrain["images"] = []
labesttrain["annotations"] = []
labelsval["info"] = labels["info"]
labelsval["licenses"] = labels["licenses"]
labelsval["categories"] = labels["categories"]
labelsval["images"] = []
labelsval["annotations"] = []
labelstest["info"] = labels["info"]
labelstest["licenses"] = labels["licenses"]
labelstest["categories"] = labels["categories"]
labelstest["images"] = []
labelstest["annotations"] = []

for index, row in train_df.iterrows():
    imagename = f"{row['patient_id']}@{row['image_id']}.png"
    imageid = -1
    for image in labels["images"]:
        if imagename == image["file_name"]:
            imageid = image["id"]
            labesttrain["images"].append(image)
            break
    for annotaion in labels["annotations"]:
        if imageid == annotaion["image_id"]:
            labesttrain["annotations"].append(annotaion)
            break
    shutil.copy2(caminhoimages + f"/{imagename}", dirtrainimage+"/images/")
jsonEncoded = json.dumps(labesttrain)
with open(f"{dirtrainimage}/labels.json", "w") as outputFile:
    outputFile.write(jsonEncoded)


for index, row in validate_df.iterrows():
    imagename = f"{row['patient_id']}@{row['image_id']}.png"
    imageid = -1
    for image in labels["images"]:
        if imagename == image["file_name"]:
            imageid = image["id"]
            labelsval["images"].append(image)
            break
    for annotaion in labels["annotations"]:
        if imageid == annotaion["image_id"]:
            labelsval["annotations"].append(annotaion)
            break
    shutil.copy2(caminhoimages + f"/{imagename}", dirvalidateimage+"/images/")
jsonEncoded = json.dumps(labelsval)
with open(f"{dirvalidateimage}/labels.json", "w") as outputFile:
    outputFile.write(jsonEncoded)

for index, row in test_df.iterrows():
    imagename = f"{row['patient_id']}@{row['image_id']}.png"
    imageid = -1
    for image in labels["images"]:
        if imagename == image["file_name"]:
            imageid = image["id"]
            labelstest["images"].append(image)
            break
    for annotaion in labels["annotations"]:
        if imageid == annotaion["image_id"]:
            labelstest["annotations"].append(annotaion)
            break
    shutil.copy2(caminhoimages + f"/{imagename}", dirtestimage+"/images/")
jsonEncoded = json.dumps(labelstest)
with open(f"{dirtestimage}/labels.json", "w") as outputFile:
    outputFile.write(jsonEncoded)
