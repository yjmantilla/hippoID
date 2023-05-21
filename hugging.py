from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageFont, ImageDraw 
import requests
import os
import sys
import time
import copy
import torchvision
sys.path.insert(0, os.path.abspath('..'))
from hippodataset import HippoDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/hippos") # you can change the folder to compare different models
TENSORBOARD = True

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms_ = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=MEAN, std=STD)
])

# Dataset
train_dataset = HippoDataset(split='train',transform=transforms_)
valid_dataset = HippoDataset(split='valid',transform=transforms_)
test_dataset = HippoDataset(split='test',transform=transforms_)
batch_size=2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=train_dataset.collate_fn)
test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True,collate_fn=test_dataset.collate_fn)
valid_dataloader =  DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,collate_fn=valid_dataset.collate_fn)

images, boxes, labels = next(iter(train_dataloader)) 

if TENSORBOARD:
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('IMAGES',img_grid)
    writer.close()

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(r'data\hippopotamus-1\valid\images\9_PNG.rf.b1da11676c58edc915954d544e5d178b.jpg')

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",return_dict=False) #https://github.com/huggingface/transformers/issues/9095

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)


def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)
        
        # Each Epoch has a training and validation phase
        for phase in ['train','val']:
            if phase=='train':
                model.train() # Set model to training mode
            else:
                model.eval()
            
            running_loss= 0.0
            running_corrects = 0

            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward
                # track history only if in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss+= loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()/dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print('')
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if TENSORBOARD:
    writer.add_graph(model,images) # Reshape as you do for the model input
    writer.close()

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
img_draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )
    if model.config.id2label[label.item()] == 'elephant':
        img_draw.rectangle(((box[0], box[1]),(box[2], box[3])), outline='Red')
    else:
        img_draw.rectangle(((box[0], box[1]),(box[2], box[3])), outline='Blue')
image.show()

model.to(device)

lr_backbone=1e-5
param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": lr_backbone,
    },
]

lr = 1e-4
lr_drop=200
weight_decay=1e-4
clip_max_norm=0.1
optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_drop)
N_EPOCHS = 2
# for epoch in range(N_EPOCHS):
#         train_stats = train_one_epoch(
#             model, criterion, data_loader_train, optimizer, device, epoch,
#             args.clip_max_norm)
#         lr_scheduler.step()
#         if args.output_dir:
#             checkpoint_paths = [output_dir / 'checkpoint.pth']
#             # extra checkpoint before LR drop and every 100 epochs
#             if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
#                 checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
#             for checkpoint_path in checkpoint_paths:
#                 utils.save_on_master({
#                     'model': model_without_ddp.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'lr_scheduler': lr_scheduler.state_dict(),
#                     'epoch': epoch,
#                     'args': args,
#                 }, checkpoint_path)

#         test_stats, coco_evaluator = evaluate(
#             model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
#         )

#         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
#                      **{f'test_{k}': v for k, v in test_stats.items()},
#                      'epoch': epoch,
#                      'n_parameters': n_parameters}

#         if args.output_dir and utils.is_main_process():
#             with (output_dir / "log.txt").open("a") as f:
#                 f.write(json.dumps(log_stats) + "\n")

#             # for evaluation logs
#             if coco_evaluator is not None:
#                 (output_dir / 'eval').mkdir(exist_ok=True)
#                 if "bbox" in coco_evaluator.coco_eval:
#                     filenames = ['latest.pth']
#                     if epoch % 50 == 0:
#                         filenames.append(f'{epoch:03}.pth')
#                     for name in filenames:
#                         torch.save(coco_evaluator.coco_eval["bbox"].eval,
#                                    output_dir / "eval" / name)

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Training time {}'.format(total_time_str))
