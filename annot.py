import json
import glob
import os
import pandas as pd
import numpy as np
root_data = r"Y:\code\hippoID\data\HIPPOPOTAMUSMAIN-12-coco"#Y:/code/hippoID/data/hippo-pretrain-1"

jsons = glob.glob(os.path.join(root_data,'**/*.json'),recursive=True)
modalities = [x.split('\\')[1] for x in jsons]


def load_json(f):
    with open(f) as fi:
        data = json.load(fi)
    return data

all_imgs = []
by_set = {}
wrong=0
for jsonpath,modality in zip(jsons,modalities):
    print(modality)
    by_set[modality]=[]
    data = load_json(jsonpath)

    #data.keys()
    cats = {x['id']:x['name'] for x in data['categories']}
    images = data['images']
    annots = data['annotations']

    for im in images:
        img_id = im['id']
        image_path  = os.path.join(root_data,modality,im['file_name'])
        valid_annots = [an for an in annots if an['image_id']==img_id]
        for an in valid_annots:
            bbox = an['bbox']
            bbox=[int(np.round(x)) for x in bbox]
            xs=[bbox[0],bbox[2]]
            ys=[bbox[1],bbox[3]]
            if len(set(xs))==1 or len(set(ys))==1 or xs[0] >= xs[1] or ys[0] >= ys[1]:
                print('same or wrong')
                print(image_path)
                wrong+=1
                print(xs,ys)
                continue
            #{path/to/image.jpg,x1,y1,x2,y2,class_name}
            all_imgs.append({'path':image_path.replace('\\','/'),'x1':min(xs),'y1':min(ys),'x2':max(xs),'y2':max(ys),'class_name':cats[an['category_id']]})
            by_set[modality].append(all_imgs[-1])

print('wrong',wrong)
df = pd.DataFrame(all_imgs)
df.to_csv("hippos_detection.csv",header=False, index=False)

#class csv
cla=[]
for i,c in enumerate(df['class_name'].unique()):
    print(c)
    cla.append({'class_name':c,'id':i})

df_cla = pd.DataFrame(cla)

df_cla.to_csv("hippos_detection_classes.csv",header=False, index=False)
for mod,l in by_set.items():
    df = pd.DataFrame(l)
    df.to_csv(f"hippos_detection_{mod}.csv",header=False, index=False)


root_data = "Y:/code/hippoID/data/hippo-pretrain-2-retina"

jsons = glob.glob(os.path.join(root_data,'**/*.csv'),recursive=True)
modalities = [x.split('\\')[1] for x in jsons]

dfs = [pd.read_csv(i.replace('\\','/'),header=None,names=['path','x1','y1','x2','y2','label']) for i in jsons]
for df,mod in zip(dfs,modalities):
    df['path']=df['path'].apply(lambda x : os.path.join(root_data,mod,x).replace('\\','/'))
    df['label']=df['label'].apply(lambda x: 'hippo')
    df.to_csv(f"hippos_detection_{mod}.csv",header=False, index=False)
    print(df.shape)

df = pd.concat(dfs)

df.to_csv("hippos_detection.csv",header=False, index=False)

assert len(df[df['x1']>=df['x2']])==0
assert len(df[df['y1']>=df['y2']])==0

cla=[]
for i,c in enumerate(df['label'].unique()):
    print(c)
    cla.append({'label':c,'id':i})

df_cla = pd.DataFrame(cla)

df_cla.to_csv("hippos_detection_classes.csv",header=False, index=False)



root_data = "Y:/code/hippoID/data/hippoID-8" # important to make it /

jsons = glob.glob(os.path.join(root_data,'**/*.csv'),recursive=True)
modalities = [x.split('\\')[1] for x in jsons]

dfs = [pd.read_csv(i.replace('\\','/'),header=None,names=['path','x1','y1','x2','y2','label']) for i in jsons]
for df,mod in zip(dfs,modalities):
    df['path']=df['path'].apply(lambda x : os.path.join(root_data,mod,x).replace('\\','/'))
    #df['label']=df['label'].apply(lambda x: 'hippo')
    df.to_csv(f"hippos_detection_{mod}.csv",header=False, index=False)
    print(df.shape)

df = pd.concat(dfs)

df.to_csv("hippos_id.csv",header=False, index=False)

assert len(df[df['x1']>=df['x2']])==0
assert len(df[df['y1']>=df['y2']])==0

cla=[]
for i,c in enumerate(df['label'].unique()):
    print(c)
    cla.append({'label':c,'id':i})

df_cla = pd.DataFrame(cla)

df_cla.to_csv("hippos_id_classes.csv",header=False, index=False)
