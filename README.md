# Projeto Final - Modelos Preditivos Conexionistas

### Joao Guilherme Bezerra Alves

|**Tipo de Projeto**|**Modelo Selecionado**|**Linguagem**|
|--|--|--|
|Deteção de Objetos|YOLOv5|PyTorch|

## Performance

O modelo treinado possui performance de **94%**.

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```text
    wandb: Currently logged in as: jgba. Use `wandb login --relogin` to force relogin
train: weights=yolov5s.pt, cfg=, data=/content/yolov5/yolov5/kart_plates-1/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=4000, batch_size=30, imgsz=305, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.2-225-g02b8a4c Python-3.7.15 torch-1.12.1+cu113 CUDA:0 (Tesla T4, 15110MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: Tracking run with wandb version 0.13.5
wandb: Run data is saved locally in /content/yolov5/yolov5/wandb/run-20221103_231305-i2cnnlrc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run devoted-lion-4
wandb: ⭐️ View project at https://wandb.ai/jgba/YOLOv5
wandb: 🚀 View run at https://wandb.ai/jgba/YOLOv5/runs/i2cnnlrc
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
100% 755k/755k [00:00<00:00, 146MB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:00<00:00, 202MB/s]

Overriding model.yaml nc=80 with nc=4

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     24273  models.yolo.Detect                      [4, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7030417 parameters, 7030417 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
WARNING ⚠️ --img-size 305 must be multiple of max stride 32, updating to 320
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.00046875), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning '/content/yolov5/yolov5/kart_plates-1/train/labels' images and labels...213 found, 0 missing, 0 empty, 0 corrupt: 100% 213/213 [00:00<00:00, 2134.76it/s]
train: New cache created: /content/yolov5/yolov5/kart_plates-1/train/labels.cache
train: Caching images (0.1GB ram): 100% 213/213 [00:01<00:00, 177.28it/s]
val: Scanning '/content/yolov5/yolov5/kart_plates-1/valid/labels' images and labels...30 found, 0 missing, 0 empty, 0 corrupt: 100% 30/30 [00:00<00:00, 767.21it/s]
val: New cache created: /content/yolov5/yolov5/kart_plates-1/valid/labels.cache
val: Caching images (0.0GB ram): 100% 30/30 [00:00<00:00, 63.41it/s]

AutoAnchor: 2.23 anchors/target, 0.956 Best Possible Recall (BPR). Anchors are a poor fit to dataset ⚠️, attempting to improve...
AutoAnchor: WARNING ⚠️ Extremely small objects found: 28 of 342 labels are <3 pixels in size
AutoAnchor: Running kmeans for 9 anchors on 342 points...
AutoAnchor: Evolving anchors with Genetic Algorithm: fitness = 0.8796: 100% 1000/1000 [00:00<00:00, 3142.95it/s]
AutoAnchor: thr=0.25: 1.0000 best possible recall, 7.68 anchors past thr
AutoAnchor: n=9, img_size=320, metric_all=0.508/0.880-mean/best, past_thr=0.566-mean: 3,5, 4,7, 5,8, 8,10, 9,13, 10,15, 12,16, 19,28, 30,37
AutoAnchor: Done ✅ (optional: update model *.yaml to use these anchors in the future)
Plotting labels to runs/train/exp/labels.jpg... 
Image sizes 320 train, 320 val
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 4000 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     0/3999      1.79G     0.1729   0.007017     0.0479         52        320:   0% 0/8 [00:01<?, ?it/s]WARNING ⚠️ TensorBoard graph visualization failure Sizes of tensors must match except in dimension 1. Expected size 20 but got size 19 for tensor number 1 in the list.
     0/3999      1.85G     0.1714   0.007239    0.04683          5        320: 100% 8/8 [00:03<00:00,  2.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.06it/s]
                   all         30         49          0          0          0          0

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     1/3999      1.85G     0.1617    0.00775    0.04439         10        320: 100% 8/8 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.00it/s]
                   all         30         49          0          0          0          0

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     2/3999      1.85G     0.1508   0.007526    0.04082          7        320: 100% 8/8 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.03it/s]
                   all         30         49          0          0          0          0

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     3/3999      1.85G     0.1432   0.007755    0.03462          8        320: 100% 8/8 [00:01<00:00,  5.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.85it/s]
                   all         30         49   0.000312     0.0625    0.00444   0.000473

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     4/3999      1.85G     0.1408   0.007722     0.0266          6        320: 100% 8/8 [00:01<00:00,  5.94it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.84it/s]
                   all         30         49    0.00144      0.256    0.00568    0.00107

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     5/3999      1.85G     0.1391   0.007908    0.01903          5        320: 100% 8/8 [00:01<00:00,  4.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.21it/s]
                   all         30         49    0.00218      0.331    0.00986    0.00241

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     6/3999      1.85G     0.1362   0.008638    0.01416          9        320: 100% 8/8 [00:01<00:00,  4.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.99it/s]
                   all         30         49      0.517     0.0625     0.0248    0.00898

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     7/3999      1.85G     0.1352   0.008582   0.008277          1        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.90it/s]
                   all         30         49      0.516      0.156     0.0701     0.0143

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     8/3999      1.85G     0.1296   0.009575   0.008994          7        320: 100% 8/8 [00:01<00:00,  6.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.55it/s]
                   all         30         49      0.396      0.175      0.141     0.0383

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     9/3999      1.85G     0.1274   0.009339   0.006811          9        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.21it/s]
                   all         30         49      0.469      0.235      0.245     0.0533

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    10/3999      1.85G     0.1254   0.008323   0.008224          5        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.49it/s]
                   all         30         49      0.628      0.275      0.137     0.0348

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    11/3999      1.85G      0.124   0.008328   0.004937          5        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.52it/s]
                   all         30         49      0.733      0.264       0.26     0.0616

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    12/3999      1.85G      0.121   0.009623   0.004843         10        320: 100% 8/8 [00:01<00:00,  6.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.42it/s]
                   all         30         49      0.793      0.355      0.336      0.114

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    13/3999      1.85G     0.1167   0.008351   0.004876          3        320: 100% 8/8 [00:01<00:00,  6.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.83it/s]
                   all         30         49      0.627      0.347      0.197     0.0673

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    14/3999      1.85G     0.1139     0.0096   0.004467          6        320: 100% 8/8 [00:01<00:00,  6.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.92it/s]
                   all         30         49      0.199      0.516      0.237     0.0711

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    15/3999      1.85G     0.1094    0.01037   0.004715          6        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.11it/s]
                   all         30         49      0.606      0.475      0.328     0.0849

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    16/3999      1.85G      0.108    0.01065   0.005466          7        320: 100% 8/8 [00:01<00:00,  6.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.24it/s]
                   all         30         49      0.608      0.465      0.399      0.124

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    17/3999      1.85G     0.1042    0.01016   0.005613          6        320: 100% 8/8 [00:01<00:00,  6.44it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.30it/s]
                   all         30         49      0.337      0.516       0.42      0.129

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    18/3999      1.85G    0.09865     0.0103   0.004933          5        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.32it/s]
                   all         30         49       0.39      0.406      0.383      0.135

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    19/3999      1.85G    0.09537    0.01124     0.0045          6        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.90it/s]
                   all         30         49      0.334      0.609      0.467      0.178

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    20/3999      1.85G    0.09165    0.01144   0.004373          5        320: 100% 8/8 [00:01<00:00,  6.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.24it/s]
                   all         30         49      0.562      0.391      0.431       0.16

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    21/3999      1.85G    0.09427    0.01058     0.0046          7        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.53it/s]
                   all         30         49       0.63      0.502      0.593      0.198

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    22/3999      1.85G    0.09251    0.01114   0.003815          5        320: 100% 8/8 [00:01<00:00,  6.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.29it/s]
                   all         30         49       0.43      0.646      0.553      0.175

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    23/3999      1.85G    0.08809    0.00966   0.004148          3        320: 100% 8/8 [00:01<00:00,  6.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.85it/s]
                   all         30         49      0.793      0.491      0.598       0.18

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    24/3999      1.85G    0.08555    0.01053   0.004007          7        320: 100% 8/8 [00:01<00:00,  6.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.92it/s]
                   all         30         49      0.713      0.531      0.578      0.205

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    25/3999      1.85G    0.08179    0.01052   0.004026          5        320: 100% 8/8 [00:01<00:00,  4.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.04it/s]
                   all         30         49      0.341      0.784      0.565      0.162

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    26/3999      1.85G    0.08683    0.01037   0.004596          8        320: 100% 8/8 [00:02<00:00,  3.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.04it/s]
                   all         30         49      0.439      0.698      0.575       0.17

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    27/3999      1.85G    0.08266   0.009822   0.004433          3        320: 100% 8/8 [00:01<00:00,  5.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.40it/s]
                   all         30         49      0.426      0.631      0.586      0.155

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    28/3999      1.85G    0.08082    0.01049   0.003924          4        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.30it/s]
                   all         30         49      0.533      0.726      0.705      0.217

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    29/3999      1.85G    0.08037    0.01014   0.004813          5        320: 100% 8/8 [00:01<00:00,  5.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.33it/s]
                   all         30         49      0.603      0.696      0.672      0.284

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    30/3999      1.85G    0.07796    0.01018   0.003575          7        320: 100% 8/8 [00:01<00:00,  5.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.511      0.773      0.664      0.254

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    31/3999      1.85G    0.08011    0.01032   0.003691          9        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.71it/s]
                   all         30         49       0.61      0.634      0.672      0.275

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    32/3999      1.85G    0.07544    0.01054    0.00349          7        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.31it/s]
                   all         30         49      0.592      0.766      0.742      0.316

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    33/3999      1.85G    0.07838   0.009129   0.003133          5        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.93it/s]
                   all         30         49       0.63      0.844      0.831      0.367

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    34/3999      1.85G     0.0737    0.01008   0.003763          6        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.87it/s]
                   all         30         49      0.551      0.754      0.715      0.315

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    35/3999      1.85G    0.07293   0.009681   0.003701          7        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.96it/s]
                   all         30         49      0.552      0.742      0.716      0.302

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    36/3999      1.85G    0.07547   0.009853   0.003542          3        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.34it/s]
                   all         30         49      0.596      0.708      0.699      0.304

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    37/3999      1.85G    0.07342    0.01055   0.003258         16        320: 100% 8/8 [00:01<00:00,  6.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.40it/s]
                   all         30         49      0.627      0.725      0.752      0.303

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    38/3999      1.85G     0.0751   0.008863   0.003062          6        320: 100% 8/8 [00:01<00:00,  6.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.16it/s]
                   all         30         49      0.602      0.771      0.724      0.338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    39/3999      1.85G    0.07039    0.01037   0.003384         10        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.28it/s]
                   all         30         49      0.515      0.785      0.772      0.333

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    40/3999      1.85G    0.06939   0.009177   0.003099          7        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.45it/s]
                   all         30         49       0.52      0.768      0.688       0.27

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    41/3999      1.85G    0.07095   0.009359    0.00305          8        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.42it/s]
                   all         30         49      0.515      0.926      0.871      0.393

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    42/3999      1.85G    0.06874   0.009136   0.003546          2        320: 100% 8/8 [00:01<00:00,  6.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.98it/s]
                   all         30         49      0.606      0.801      0.855      0.396

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    43/3999      1.85G    0.07263    0.00934   0.003844          8        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.35it/s]
                   all         30         49      0.787      0.686      0.837      0.358

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    44/3999      1.85G    0.06899   0.009515   0.003203          8        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.40it/s]
                   all         30         49      0.831      0.719      0.841      0.368

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    45/3999      1.85G    0.06767   0.009059   0.004231          5        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.02it/s]
                   all         30         49      0.685      0.725      0.795       0.29

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    46/3999      1.85G     0.0715   0.008944   0.003587          4        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.23it/s]
                   all         30         49      0.775       0.76      0.831      0.343

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    47/3999      1.85G    0.06825    0.00901   0.003105          6        320: 100% 8/8 [00:01<00:00,  6.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.791      0.777      0.812       0.35

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    48/3999      1.85G    0.06921   0.008709   0.002743          6        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.60it/s]
                   all         30         49      0.659      0.885      0.838      0.354

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    49/3999      1.85G    0.06543   0.009688    0.00252          6        320: 100% 8/8 [00:01<00:00,  6.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.45it/s]
                   all         30         49      0.795      0.788      0.855      0.349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    50/3999      1.85G    0.06402   0.008938   0.002918          5        320: 100% 8/8 [00:02<00:00,  3.93it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.44it/s]
                   all         30         49      0.673      0.755      0.721      0.293

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    51/3999      1.85G    0.06616   0.008956   0.003277          5        320: 100% 8/8 [00:02<00:00,  3.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.05it/s]
                   all         30         49       0.78      0.807      0.833      0.316

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    52/3999      1.85G    0.06649   0.007864   0.003305          4        320: 100% 8/8 [00:02<00:00,  3.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.25it/s]
                   all         30         49      0.868      0.788      0.901      0.436

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    53/3999      1.85G    0.06633   0.008701   0.003078          7        320: 100% 8/8 [00:01<00:00,  5.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.19it/s]
                   all         30         49      0.799      0.809      0.911      0.465

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    54/3999      1.85G    0.06537   0.008231   0.004675          2        320: 100% 8/8 [00:01<00:00,  6.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.45it/s]
                   all         30         49      0.762      0.821      0.878      0.388

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    55/3999      1.85G    0.06604   0.009035   0.002841          7        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.28it/s]
                   all         30         49      0.843      0.834      0.891      0.431

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    56/3999      1.85G    0.06591   0.008914   0.002972          7        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.28it/s]
                   all         30         49      0.808      0.792      0.859      0.369

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    57/3999      1.85G    0.06545   0.009725    0.00276          9        320: 100% 8/8 [00:01<00:00,  6.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.09it/s]
                   all         30         49      0.852      0.865      0.941      0.423

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    58/3999      1.85G    0.06869   0.009574   0.002975          9        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.30it/s]
                   all         30         49      0.814      0.865      0.914      0.423

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    59/3999      1.85G    0.06603   0.009856   0.002477         10        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.98it/s]
                   all         30         49      0.855      0.868      0.906      0.457

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    60/3999      1.85G    0.06286   0.007907   0.002917          2        320: 100% 8/8 [00:01<00:00,  6.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.81it/s]
                   all         30         49      0.815      0.775      0.887      0.429

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    61/3999      1.85G    0.06186    0.00886   0.003161          7        320: 100% 8/8 [00:01<00:00,  6.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.08it/s]
                   all         30         49      0.728      0.933      0.931      0.459

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    62/3999      1.85G    0.06492   0.008357    0.00281          5        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.00it/s]
                   all         30         49      0.773      0.846      0.879      0.491

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    63/3999      1.85G    0.06171   0.008641   0.002662          7        320: 100% 8/8 [00:01<00:00,  6.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.777      0.882      0.888      0.406

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    64/3999      1.85G    0.06461    0.00871    0.00259          5        320: 100% 8/8 [00:01<00:00,  6.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.36it/s]
                   all         30         49       0.68      0.805      0.845      0.417

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    65/3999      1.85G    0.06407   0.008595   0.002578          8        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.33it/s]
                   all         30         49      0.648      0.894      0.876      0.395

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    66/3999      1.85G    0.06525   0.009285   0.002727         12        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.94it/s]
                   all         30         49      0.777      0.823      0.861      0.429

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    67/3999      1.85G    0.06056   0.008589   0.002506          4        320: 100% 8/8 [00:01<00:00,  6.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.79it/s]
                   all         30         49      0.771      0.863      0.825       0.43

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    68/3999      1.85G     0.0604   0.008153   0.002873          4        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.30it/s]
                   all         30         49      0.907      0.855      0.943      0.454

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    69/3999      1.85G    0.06029   0.008895   0.002423          6        320: 100% 8/8 [00:01<00:00,  6.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.10it/s]
                   all         30         49      0.829      0.891      0.904      0.438

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    70/3999      1.85G    0.06274   0.009106   0.002691          8        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.25it/s]
                   all         30         49      0.782      0.909      0.906       0.41

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    71/3999      1.85G    0.06122   0.008455   0.002943          8        320: 100% 8/8 [00:01<00:00,  6.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.40it/s]
                   all         30         49       0.87      0.864      0.894       0.44

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    72/3999      1.85G    0.06174   0.008987   0.002529         10        320: 100% 8/8 [00:01<00:00,  6.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.44it/s]
                   all         30         49      0.827      0.808        0.9      0.483

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    73/3999      1.85G    0.06094   0.007796   0.002463          3        320: 100% 8/8 [00:01<00:00,  6.43it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.911      0.774      0.878      0.404

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    74/3999      1.85G    0.05835   0.007926   0.002454          2        320: 100% 8/8 [00:01<00:00,  6.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.19it/s]
                   all         30         49      0.746      0.851      0.841       0.45

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    75/3999      1.85G    0.06005   0.007994   0.002729          6        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.12it/s]
                   all         30         49      0.764      0.875      0.927      0.459

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    76/3999      1.85G    0.05629   0.007937   0.002332          4        320: 100% 8/8 [00:01<00:00,  6.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.70it/s]
                   all         30         49      0.794      0.878      0.927      0.481

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    77/3999      1.85G    0.06134   0.008551   0.002303          9        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.34it/s]
                   all         30         49      0.905      0.821      0.914      0.457

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    78/3999      1.85G    0.05732   0.008804   0.002474          6        320: 100% 8/8 [00:01<00:00,  6.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.60it/s]
                   all         30         49      0.826      0.866      0.917       0.38

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    79/3999      1.85G     0.0595   0.008373   0.002462          6        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.36it/s]
                   all         30         49      0.838      0.868      0.903      0.422

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    80/3999      1.85G    0.06041   0.008151   0.002372          6        320: 100% 8/8 [00:01<00:00,  6.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.53it/s]
                   all         30         49      0.829       0.82      0.846      0.423

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    81/3999      1.85G    0.05797   0.008214   0.002265          5        320: 100% 8/8 [00:01<00:00,  6.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.65it/s]
                   all         30         49      0.879       0.85      0.916       0.44

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    82/3999      1.85G    0.05866   0.008524   0.002603          5        320: 100% 8/8 [00:01<00:00,  6.06it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.95it/s]
                   all         30         49      0.768        0.9      0.933      0.465

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    83/3999      1.85G    0.05609   0.007365   0.002355          2        320: 100% 8/8 [00:02<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.83it/s]
                   all         30         49      0.909      0.933       0.95      0.499

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    84/3999      1.85G     0.0581   0.008134   0.002399          5        320: 100% 8/8 [00:01<00:00,  4.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.91it/s]
                   all         30         49      0.855      0.901      0.922      0.474

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    85/3999      1.85G    0.05623   0.008095   0.002102          4        320: 100% 8/8 [00:01<00:00,  6.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.33it/s]
                   all         30         49      0.845      0.917      0.912      0.503

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    86/3999      1.85G    0.05775   0.008444   0.002285          7        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.81it/s]
                   all         30         49      0.854      0.891      0.905      0.466

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    87/3999      1.85G    0.05924   0.008044   0.002464          6        320: 100% 8/8 [00:01<00:00,  6.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.15it/s]
                   all         30         49       0.89      0.838      0.901      0.481

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    88/3999      1.85G    0.05582   0.008002   0.002957          4        320: 100% 8/8 [00:01<00:00,  6.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.29it/s]
                   all         30         49      0.906      0.842      0.906      0.486

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    89/3999      1.85G    0.05624    0.00808   0.002638          5        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.60it/s]
                   all         30         49      0.954       0.87      0.932      0.481

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    90/3999      1.85G    0.05523   0.007212   0.002073          1        320: 100% 8/8 [00:01<00:00,  6.15it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.70it/s]
                   all         30         49       0.86      0.861      0.911      0.466

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    91/3999      1.85G    0.05744    0.00879   0.001949          9        320: 100% 8/8 [00:01<00:00,  6.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.34it/s]
                   all         30         49      0.807      0.883      0.919      0.424

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    92/3999      1.85G    0.05876   0.008077   0.002154          8        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.08it/s]
                   all         30         49      0.795      0.902      0.876      0.415

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    93/3999      1.85G    0.06046   0.008307   0.002211          7        320: 100% 8/8 [00:01<00:00,  6.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.12it/s]
                   all         30         49      0.766      0.918       0.87       0.39

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    94/3999      1.85G    0.05744   0.008113   0.001954          6        320: 100% 8/8 [00:01<00:00,  6.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.809      0.918      0.908      0.438

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    95/3999      1.85G    0.05632   0.008373   0.002074          7        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.67it/s]
                   all         30         49      0.888      0.917      0.968      0.489

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    96/3999      1.85G    0.05425   0.007128   0.002197          2        320: 100% 8/8 [00:01<00:00,  6.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.16it/s]
                   all         30         49      0.749      0.983      0.932      0.465

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    97/3999      1.85G    0.05737   0.008473   0.002214          8        320: 100% 8/8 [00:01<00:00,  6.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.815      0.931      0.934      0.498

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    98/3999      1.85G     0.0553   0.008206   0.002084          9        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.46it/s]
                   all         30         49      0.836      0.965      0.959      0.507

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    99/3999      1.85G    0.05556   0.007916   0.002342          5        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.60it/s]
                   all         30         49      0.891       0.82      0.906      0.449

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   100/3999      1.85G    0.05643   0.007834   0.002791          7        320: 100% 8/8 [00:01<00:00,  5.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.884      0.775      0.893      0.462

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   101/3999      1.85G    0.05934   0.007647   0.003124          5        320: 100% 8/8 [00:01<00:00,  6.43it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.22it/s]
                   all         30         49      0.832      0.871      0.911      0.419

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   102/3999      1.85G    0.05478   0.008491   0.002539          8        320: 100% 8/8 [00:01<00:00,  6.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.67it/s]
                   all         30         49      0.875      0.842      0.904      0.427

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   103/3999      1.85G    0.05654   0.007725   0.002341          3        320: 100% 8/8 [00:01<00:00,  5.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.96it/s]
                   all         30         49      0.758      0.937      0.923      0.459

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   104/3999      1.85G    0.05558   0.008136   0.002318          5        320: 100% 8/8 [00:01<00:00,  4.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.04it/s]
                   all         30         49      0.725      0.955       0.92      0.488

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   105/3999      1.85G    0.05557   0.007799   0.002236          6        320: 100% 8/8 [00:01<00:00,  4.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.58it/s]
                   all         30         49      0.736      0.884      0.913      0.506

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   106/3999      1.85G    0.05747   0.007706   0.002307          5        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.70it/s]
                   all         30         49      0.904      0.864      0.952      0.529

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   107/3999      1.85G    0.05381   0.007181   0.002276          4        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.08it/s]
                   all         30         49      0.896      0.873      0.925      0.429

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   108/3999      1.85G    0.05346   0.007217   0.002094          4        320: 100% 8/8 [00:01<00:00,  5.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.14it/s]
                   all         30         49      0.905      0.898      0.971      0.513

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   109/3999      1.85G    0.05188   0.007259   0.001988          3        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.88it/s]
                   all         30         49      0.893      0.963      0.976      0.476

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   110/3999      1.85G    0.05825   0.008079   0.002262          7        320: 100% 8/8 [00:01<00:00,  6.43it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.16it/s]
                   all         30         49      0.896      0.931      0.951      0.482

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   111/3999      1.85G    0.04984   0.007124   0.002161          1        320: 100% 8/8 [00:01<00:00,  6.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.63it/s]
                   all         30         49      0.912      0.967      0.956      0.537

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   112/3999      1.85G    0.05409    0.00782   0.002346          6        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.29it/s]
                   all         30         49       0.88      0.968      0.955      0.487

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   113/3999      1.85G    0.05213    0.00728   0.002186          2        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.17it/s]
                   all         30         49      0.902      0.951      0.963      0.466

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   114/3999      1.85G    0.05364   0.008475   0.002185          5        320: 100% 8/8 [00:01<00:00,  6.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.77it/s]
                   all         30         49      0.858       0.88      0.918       0.47

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   115/3999      1.85G    0.05529   0.007685   0.002142          4        320: 100% 8/8 [00:01<00:00,  5.94it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.56it/s]
                   all         30         49      0.795      0.818      0.874      0.455

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   116/3999      1.85G    0.05491   0.008065   0.001751          2        320: 100% 8/8 [00:01<00:00,  6.44it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.75it/s]
                   all         30         49      0.747      0.801      0.851      0.443

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   117/3999      1.85G    0.05514   0.008128   0.002127          8        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.31it/s]
                   all         30         49      0.828      0.851      0.909      0.456

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   118/3999      1.85G    0.05681   0.009002   0.002559          9        320: 100% 8/8 [00:01<00:00,  6.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.43it/s]
                   all         30         49      0.888      0.856      0.892      0.456

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   119/3999      1.85G    0.05127   0.007719    0.00232          2        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.68it/s]
                   all         30         49      0.881      0.868      0.918      0.457

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   120/3999      1.85G    0.05511   0.007911   0.002043          9        320: 100% 8/8 [00:01<00:00,  6.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.68it/s]
                   all         30         49      0.933      0.868      0.918      0.483

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   121/3999      1.85G    0.05177   0.008246    0.00172          3        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.46it/s]
                   all         30         49      0.864      0.868       0.91      0.469

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   122/3999      1.85G    0.05189   0.007679   0.001805          8        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.86it/s]
                   all         30         49      0.874      0.851      0.908      0.478

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   123/3999      1.85G    0.05436   0.008517   0.001997          8        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.07it/s]
                   all         30         49      0.908      0.888      0.937      0.512

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   124/3999      1.85G    0.05254   0.007604   0.002056          2        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.21it/s]
                   all         30         49      0.978      0.912      0.953      0.527

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   125/3999      1.85G    0.05148   0.008311   0.002135          6        320: 100% 8/8 [00:01<00:00,  6.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.23it/s]
                   all         30         49      0.886      0.883      0.919      0.535

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   126/3999      1.85G    0.05333   0.007833   0.002136          6        320: 100% 8/8 [00:01<00:00,  6.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.12it/s]
                   all         30         49       0.82      0.834      0.827      0.473

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   127/3999      1.85G    0.05316   0.007367   0.001821          5        320: 100% 8/8 [00:01<00:00,  6.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.99it/s]
                   all         30         49      0.876      0.888      0.877      0.499

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   128/3999      1.85G     0.0512   0.007737   0.001837          8        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.27it/s]
                   all         30         49      0.839        0.9      0.929      0.493

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   129/3999      1.85G    0.04955   0.007661   0.002343          3        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.30it/s]
                   all         30         49      0.888      0.943      0.967      0.503

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   130/3999      1.85G    0.05191   0.007734   0.002102          7        320: 100% 8/8 [00:01<00:00,  6.45it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.79it/s]
                   all         30         49      0.929      0.906      0.935      0.475

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   131/3999      1.85G    0.05433   0.007649   0.002052          7        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.45it/s]
                   all         30         49      0.885      0.902       0.94      0.488

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   132/3999      1.85G    0.05081   0.007634   0.002112          4        320: 100% 8/8 [00:01<00:00,  5.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.43it/s]
                   all         30         49      0.914      0.887      0.938      0.531

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   133/3999      1.85G    0.05136   0.008074   0.001937         10        320: 100% 8/8 [00:01<00:00,  5.99it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.90it/s]
                   all         30         49      0.814      0.904      0.925      0.483

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   134/3999      1.85G    0.05337   0.006897   0.001884          4        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.73it/s]
                   all         30         49      0.797      0.835      0.843      0.427

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   135/3999      1.85G    0.05507   0.007423   0.002006          4        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.05it/s]
                   all         30         49      0.883      0.912      0.915       0.46

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   136/3999      1.85G    0.05194   0.007537   0.002111          6        320: 100% 8/8 [00:01<00:00,  6.41it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.18it/s]
                   all         30         49      0.893      0.884      0.914      0.574

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   137/3999      1.85G    0.04889   0.007866   0.002049          4        320: 100% 8/8 [00:01<00:00,  6.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.11it/s]
                   all         30         49      0.849      0.917      0.926      0.518

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   138/3999      1.85G    0.04929   0.007684   0.001772          5        320: 100% 8/8 [00:01<00:00,  6.41it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.53it/s]
                   all         30         49      0.895        0.9      0.926      0.524

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   139/3999      1.85G    0.05036   0.007326   0.002186          5        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.16it/s]
                   all         30         49      0.878      0.864      0.914      0.496

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   140/3999      1.85G    0.04979   0.007164    0.00225          4        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.02it/s]
                   all         30         49       0.79      0.886      0.893      0.485

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   141/3999      1.85G    0.05259   0.008822   0.002081          8        320: 100% 8/8 [00:01<00:00,  6.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.14it/s]
                   all         30         49      0.884        0.9      0.929       0.49

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   142/3999      1.85G    0.05003   0.008275   0.002031          8        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.80it/s]
                   all         30         49      0.934      0.967      0.979      0.565

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   143/3999      1.85G    0.05017   0.007611   0.001704          6        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.00it/s]
                   all         30         49      0.897       0.92      0.957       0.56

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   144/3999      1.85G    0.05007   0.007891   0.002059          4        320: 100% 8/8 [00:01<00:00,  6.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.91it/s]
                   all         30         49      0.874      0.951      0.956      0.519

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   145/3999      1.85G    0.05038     0.0073   0.002047          5        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.22it/s]
                   all         30         49      0.863      0.918      0.935      0.501

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   146/3999      1.85G    0.05188   0.007397   0.002343          5        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.77it/s]
                   all         30         49      0.927      0.815      0.909      0.513

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   147/3999      1.85G    0.04988   0.007415   0.001859          4        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.46it/s]
                   all         30         49      0.834      0.901       0.91      0.513

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   148/3999      1.85G    0.04917   0.008059   0.001702          7        320: 100% 8/8 [00:01<00:00,  6.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.14it/s]
                   all         30         49      0.875       0.95      0.949      0.489

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   149/3999      1.85G    0.05386   0.007202   0.001729          2        320: 100% 8/8 [00:01<00:00,  5.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.92it/s]
                   all         30         49      0.868      0.871      0.938      0.481

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   150/3999      1.85G    0.05143   0.008475   0.001725          6        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.11it/s]
                   all         30         49      0.905      0.967      0.951      0.501

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   151/3999      1.85G    0.04901   0.008103   0.001756          6        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.10it/s]
                   all         30         49       0.95       0.95      0.976      0.487

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   152/3999      1.85G    0.04811   0.007167   0.002101          4        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.46it/s]
                   all         30         49       0.96      0.917      0.939      0.528

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   153/3999      1.85G    0.04986   0.007416   0.001857          4        320: 100% 8/8 [00:01<00:00,  6.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.11it/s]
                   all         30         49      0.908      0.898      0.913       0.51

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   154/3999      1.85G    0.05053   0.007652   0.001958          6        320: 100% 8/8 [00:01<00:00,  6.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.09it/s]
                   all         30         49      0.959      0.983      0.978      0.512

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   155/3999      1.85G    0.04834   0.006861   0.001994          3        320: 100% 8/8 [00:01<00:00,  6.42it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.82it/s]
                   all         30         49      0.905      0.854      0.947      0.506

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   156/3999      1.85G      0.049   0.007718   0.002025         10        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.29it/s]
                   all         30         49      0.905      0.882      0.962      0.534

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   157/3999      1.85G    0.04703   0.008425   0.001758          7        320: 100% 8/8 [00:01<00:00,  6.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.18it/s]
                   all         30         49       0.77      0.934      0.905      0.481

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   158/3999      1.85G    0.05003   0.007112   0.001849          5        320: 100% 8/8 [00:01<00:00,  5.92it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.889      0.913      0.934      0.512

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   159/3999      1.85G    0.05263   0.007831   0.001723          8        320: 100% 8/8 [00:01<00:00,  6.08it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.17it/s]
                   all         30         49       0.87      0.918      0.924      0.503

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   160/3999      1.85G    0.05106   0.007162   0.002103          5        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.98it/s]
                   all         30         49      0.879       0.92      0.906      0.483

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   161/3999      1.85G    0.04809   0.007361   0.001589          3        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.58it/s]
                   all         30         49      0.871      0.935      0.914      0.513

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   162/3999      1.85G    0.05141   0.007622   0.001701          7        320: 100% 8/8 [00:01<00:00,  6.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.62it/s]
                   all         30         49      0.843      0.951      0.937      0.514

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   163/3999      1.85G    0.04938   0.007012   0.001656          6        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.31it/s]
                   all         30         49      0.885      0.968      0.957      0.532

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   164/3999      1.85G    0.04892   0.008112   0.001755          6        320: 100% 8/8 [00:01<00:00,  6.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.42it/s]
                   all         30         49      0.828      0.961      0.956      0.543

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   165/3999      1.85G    0.05048   0.007711   0.001954          4        320: 100% 8/8 [00:01<00:00,  6.15it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.29it/s]
                   all         30         49      0.833      0.917      0.934      0.535

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   166/3999      1.85G    0.04882   0.007321   0.001697          5        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.39it/s]
                   all         30         49       0.92       0.95      0.974      0.506

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   167/3999      1.85G    0.04813   0.008742   0.002156         11        320: 100% 8/8 [00:01<00:00,  6.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.15it/s]
                   all         30         49      0.886      0.946      0.959      0.532

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   168/3999      1.85G    0.05218   0.007864   0.001803          6        320: 100% 8/8 [00:01<00:00,  6.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.24it/s]
                   all         30         49      0.973      0.913      0.952      0.505

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   169/3999      1.85G    0.04779   0.006533   0.001914          1        320: 100% 8/8 [00:01<00:00,  6.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.97it/s]
                   all         30         49      0.948      0.917      0.954      0.542

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   170/3999      1.85G    0.04796   0.007283   0.001881          3        320: 100% 8/8 [00:01<00:00,  6.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.49it/s]
                   all         30         49      0.929      0.917       0.95      0.512

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   171/3999      1.85G     0.0491   0.007701    0.00186          8        320: 100% 8/8 [00:01<00:00,  6.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.44it/s]
                   all         30         49      0.899      0.966      0.955      0.494

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   172/3999      1.85G    0.04794   0.007869   0.001731          8        320: 100% 8/8 [00:01<00:00,  6.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.04it/s]
                   all         30         49      0.903      0.911      0.931      0.508

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   173/3999      1.85G    0.04797   0.007636   0.001681          6        320: 100% 8/8 [00:01<00:00,  6.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.61it/s]
                   all         30         49      0.891      0.906      0.917      0.523

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   174/3999      1.85G    0.04966   0.008141   0.001753          9        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.76it/s]
                   all         30         49      0.894      0.912      0.933      0.496

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   175/3999      1.85G    0.05043   0.007348   0.001909          4        320: 100% 8/8 [00:01<00:00,  4.99it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.11it/s]
                   all         30         49      0.899      0.913      0.931      0.477

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   176/3999      1.85G     0.0467   0.007781   0.002074          8        320: 100% 8/8 [00:02<00:00,  3.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.49it/s]
                   all         30         49      0.924      0.917       0.93      0.527

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   177/3999      1.85G     0.0459    0.00654   0.002195          4        320: 100% 8/8 [00:02<00:00,  3.47it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.71it/s]
                   all         30         49      0.944      0.911       0.93      0.498

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   178/3999      1.85G    0.04625   0.008199   0.001756         10        320: 100% 8/8 [00:02<00:00,  3.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.62it/s]
                   all         30         49      0.922      0.869      0.921      0.514

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   179/3999      1.85G    0.05158   0.007588   0.001783          5        320: 100% 8/8 [00:02<00:00,  3.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.62it/s]
                   all         30         49      0.898       0.96      0.961      0.523

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   180/3999      1.85G    0.04761   0.007707   0.001547          6        320: 100% 8/8 [00:01<00:00,  6.57it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.31it/s]
                   all         30         49      0.855      0.949      0.947      0.532

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   181/3999      1.85G    0.04836   0.007764   0.001596          7        320: 100% 8/8 [00:01<00:00,  6.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.84it/s]
                   all         30         49      0.841      0.921      0.944      0.527

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   182/3999      1.85G    0.04696   0.007893    0.00187          5        320: 100% 8/8 [00:02<00:00,  3.94it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.17it/s]
                   all         30         49      0.852      0.955      0.938      0.562

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   183/3999      1.85G    0.04702   0.007738   0.001737          4        320: 100% 8/8 [00:02<00:00,  3.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.98it/s]
                   all         30         49      0.955      0.917       0.96      0.562

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   184/3999      1.85G    0.04574   0.007189   0.001865          5        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.76it/s]
                   all         30         49      0.949      0.925       0.97      0.525

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   185/3999      1.85G     0.0483   0.007998   0.001957          8        320: 100% 8/8 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.43it/s]
                   all         30         49      0.884      0.911      0.963      0.563

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   186/3999      1.85G    0.04675   0.007199   0.001977          2        320: 100% 8/8 [00:01<00:00,  6.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.12it/s]
                   all         30         49      0.923      0.917      0.967      0.523

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   187/3999      1.85G    0.04605   0.007436   0.002152          5        320: 100% 8/8 [00:01<00:00,  6.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.83it/s]
                   all         30         49      0.902        0.9      0.926      0.502

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   188/3999      1.85G    0.04723   0.007845   0.001701         10        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.18it/s]
                   all         30         49      0.875       0.95      0.926      0.478

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   189/3999      1.85G    0.04938   0.006619   0.002422          2        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.22it/s]
                   all         30         49      0.861      0.935      0.938      0.504

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   190/3999      1.85G    0.04899    0.00767   0.001754          4        320: 100% 8/8 [00:01<00:00,  6.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.19it/s]
                   all         30         49      0.912       0.95      0.971      0.569

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   191/3999      1.85G    0.04519   0.006884   0.001725          1        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.27it/s]
                   all         30         49       0.91      0.938      0.965      0.581

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   192/3999      1.85G    0.04722   0.007163   0.002414          3        320: 100% 8/8 [00:01<00:00,  6.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.20it/s]
                   all         30         49      0.916      0.907      0.936      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   193/3999      1.85G    0.04744   0.007369   0.001846          5        320: 100% 8/8 [00:01<00:00,  6.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.50it/s]
                   all         30         49      0.942      0.963      0.978      0.549

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   194/3999      1.85G      0.045   0.006778   0.002134          2        320: 100% 8/8 [00:01<00:00,  5.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.43it/s]
                   all         30         49      0.951      0.933      0.953      0.566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   195/3999      1.85G    0.04527   0.007552   0.007961          3        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.86it/s]
                   all         30         49      0.882      0.938      0.947      0.557

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   196/3999      1.85G    0.04936   0.007708   0.002008          9        320: 100% 8/8 [00:01<00:00,  5.93it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.02it/s]
                   all         30         49      0.906      0.943      0.939       0.54

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   197/3999      1.85G     0.0459   0.007629   0.001657          6        320: 100% 8/8 [00:01<00:00,  6.05it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.20it/s]
                   all         30         49      0.885      0.919      0.935      0.538

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   198/3999      1.85G    0.04697   0.007604   0.001683          8        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.87it/s]
                   all         30         49      0.825      0.869      0.895      0.526

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   199/3999      1.85G    0.04627   0.007793   0.001552          7        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.97it/s]
                   all         30         49      0.881      0.897       0.92      0.522

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   200/3999      1.85G    0.05115    0.00782   0.001711          8        320: 100% 8/8 [00:01<00:00,  6.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.48it/s]
                   all         30         49      0.863      0.869      0.914       0.54

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   201/3999      1.85G    0.04668   0.007803   0.001628         11        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.10it/s]
                   all         30         49       0.86      0.919      0.918      0.493

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   202/3999      1.85G    0.04672   0.007632   0.001653          7        320: 100% 8/8 [00:01<00:00,  6.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.99it/s]
                   all         30         49      0.947      0.901      0.952      0.517

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   203/3999      1.85G    0.04678   0.007557   0.001566          9        320: 100% 8/8 [00:01<00:00,  6.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.72it/s]
                   all         30         49      0.874      0.933      0.964      0.529

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   204/3999      1.85G    0.04717   0.007809   0.001735          7        320: 100% 8/8 [00:01<00:00,  6.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.01it/s]
                   all         30         49      0.973      0.915      0.966       0.55

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   205/3999      1.85G    0.04622    0.00788   0.001622          4        320: 100% 8/8 [00:01<00:00,  5.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.99it/s]
                   all         30         49      0.955      0.914      0.948      0.584

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   206/3999      1.85G    0.04625   0.007222   0.001537          7        320: 100% 8/8 [00:01<00:00,  6.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.19it/s]
                   all         30         49      0.959      0.918      0.947      0.531

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   207/3999      1.85G    0.04504   0.007383   0.001553          6        320: 100% 8/8 [00:01<00:00,  6.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.88it/s]
                   all         30         49      0.925      0.924      0.945      0.544

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   208/3999      1.85G    0.04966   0.008458   0.001593         10        320: 100% 8/8 [00:01<00:00,  6.15it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.50it/s]
                   all         30         49      0.928       0.95      0.968      0.549

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   209/3999      1.85G      0.047   0.006901   0.001505          5        320: 100% 8/8 [00:01<00:00,  6.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.08it/s]
                   all         30         49      0.927      0.919      0.941       0.54

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   210/3999      1.85G    0.04857   0.006905   0.001654          5        320: 100% 8/8 [00:01<00:00,  6.44it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.07it/s]
                   all         30         49      0.967      0.936      0.974      0.561

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   211/3999      1.85G     0.0474    0.00749   0.002077          5        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.43it/s]
                   all         30         49      0.913      0.939      0.964       0.53

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   212/3999      1.85G    0.04587    0.00767   0.001842          5        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.04it/s]
                   all         30         49       0.88      0.933      0.962      0.554

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   213/3999      1.85G    0.04731   0.007657   0.002014          7        320: 100% 8/8 [00:01<00:00,  6.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.07it/s]
                   all         30         49      0.982       0.92      0.957      0.562

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   214/3999      1.85G    0.04713   0.006992   0.001789          7        320: 100% 8/8 [00:01<00:00,  5.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.97it/s]
                   all         30         49      0.928      0.978      0.978      0.557

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   215/3999      1.85G    0.04465   0.007692   0.001556          5        320: 100% 8/8 [00:01<00:00,  6.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.11it/s]
                   all         30         49      0.978      0.915      0.954      0.542

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   216/3999      1.85G     0.0467   0.007058   0.001732          5        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.96it/s]
                   all         30         49      0.981      0.947      0.974      0.527

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   217/3999      1.85G    0.04699   0.007969   0.002038          7        320: 100% 8/8 [00:01<00:00,  5.96it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.02it/s]
                   all         30         49      0.965      0.932      0.963      0.522

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   218/3999      1.85G    0.04607   0.008454    0.00151         10        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.12it/s]
                   all         30         49      0.912      0.968      0.972      0.552

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   219/3999      1.85G      0.046   0.007319   0.001956          6        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.42it/s]
                   all         30         49      0.968      0.933       0.97      0.534

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   220/3999      1.85G    0.04401   0.007062   0.001546          4        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.15it/s]
                   all         30         49      0.944      0.933      0.959      0.519

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   221/3999      1.85G    0.04392   0.006358   0.001444          2        320: 100% 8/8 [00:01<00:00,  6.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.95it/s]
                   all         30         49      0.902      0.933      0.953      0.485

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   222/3999      1.85G    0.04667   0.007104   0.001729          3        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.92it/s]
                   all         30         49      0.922       0.95      0.967       0.53

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   223/3999      1.85G    0.04713    0.00742   0.002096          6        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.38it/s]
                   all         30         49      0.898      0.932      0.943      0.571

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   224/3999      1.85G    0.04439   0.007134   0.001903          5        320: 100% 8/8 [00:01<00:00,  6.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.89it/s]
                   all         30         49      0.954      0.967      0.977      0.551

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   225/3999      1.85G    0.04748   0.008588   0.001732          7        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.00it/s]
                   all         30         49       0.91      0.933      0.959      0.564

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   226/3999      1.85G    0.04428   0.007341   0.001645          4        320: 100% 8/8 [00:01<00:00,  6.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.17it/s]
                   all         30         49      0.971      0.917       0.94      0.568

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   227/3999      1.85G    0.04407   0.006964   0.001691          5        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.60it/s]
                   all         30         49      0.984      0.929      0.968      0.551

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   228/3999      1.85G    0.04345   0.007169   0.001612          6        320: 100% 8/8 [00:01<00:00,  6.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.37it/s]
                   all         30         49      0.981      0.928      0.966       0.57

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   229/3999      1.85G      0.043   0.007032   0.001549          5        320: 100% 8/8 [00:01<00:00,  5.95it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.09it/s]
                   all         30         49      0.917       0.95      0.972      0.586

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   230/3999      1.85G    0.04261   0.007088    0.00168          4        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.96it/s]
                   all         30         49      0.941      0.983      0.975      0.581

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   231/3999      1.85G    0.04733   0.006531   0.001611          4        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.76it/s]
                   all         30         49       0.94      0.949      0.954       0.56

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   232/3999      1.85G    0.04509   0.006725   0.001585          4        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.31it/s]
                   all         30         49      0.934       0.98      0.974       0.57

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   233/3999      1.85G    0.04627   0.006642   0.001681          8        320: 100% 8/8 [00:01<00:00,  6.29it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.75it/s]
                   all         30         49       0.94      0.957      0.975      0.569

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   234/3999      1.85G    0.04448   0.006901   0.001425          3        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.68it/s]
                   all         30         49      0.891      0.927      0.956      0.563

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   235/3999      1.85G    0.04458   0.007254   0.001452          8        320: 100% 8/8 [00:01<00:00,  6.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.18it/s]
                   all         30         49      0.972      0.914      0.963      0.574

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   236/3999      1.85G    0.04332   0.006848   0.001746          2        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.14it/s]
                   all         30         49       0.97      0.915      0.951      0.556

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   237/3999      1.85G    0.04603   0.007403   0.001445          6        320: 100% 8/8 [00:01<00:00,  6.38it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.60it/s]
                   all         30         49      0.941      0.901       0.94      0.549

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   238/3999      1.85G    0.04762   0.007552   0.001966          9        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.90it/s]
                   all         30         49      0.886      0.935      0.959      0.554

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   239/3999      1.85G    0.04655   0.008032   0.001658          8        320: 100% 8/8 [00:01<00:00,  6.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.15it/s]
                   all         30         49      0.924       0.96      0.953      0.553

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   240/3999      1.85G     0.0447   0.007568   0.001517          9        320: 100% 8/8 [00:01<00:00,  6.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.56it/s]
                   all         30         49      0.883       0.95      0.937      0.551

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   241/3999      1.85G    0.04386   0.007969   0.001555         11        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.64it/s]
                   all         30         49      0.908       0.96      0.945      0.574

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   242/3999      1.85G    0.04576   0.007512   0.001294          3        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.62it/s]
                   all         30         49       0.96      0.917      0.958      0.549

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   243/3999      1.85G    0.04486   0.007238    0.00132          3        320: 100% 8/8 [00:01<00:00,  6.07it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.86it/s]
                   all         30         49      0.869      0.947      0.949      0.556

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   244/3999      1.85G    0.04342   0.007808   0.001612          3        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.22it/s]
                   all         30         49       0.96      0.933      0.974      0.566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   245/3999      1.85G    0.04568   0.008279   0.001958          7        320: 100% 8/8 [00:01<00:00,  5.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.08it/s]
                   all         30         49      0.926      0.979      0.977       0.55

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   246/3999      1.85G    0.04318   0.007155    0.00163          3        320: 100% 8/8 [00:01<00:00,  6.05it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.61it/s]
                   all         30         49       0.95      0.917      0.943      0.544

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   247/3999      1.85G    0.04286   0.006834   0.001635          3        320: 100% 8/8 [00:01<00:00,  6.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.73it/s]
                   all         30         49      0.916      0.941      0.927       0.55

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   248/3999      1.85G    0.04327   0.007005   0.001612          4        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.18it/s]
                   all         30         49      0.911      0.883      0.925      0.557

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   249/3999      1.85G    0.04494   0.006948   0.001354          4        320: 100% 8/8 [00:01<00:00,  6.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.33it/s]
                   all         30         49      0.942      0.933      0.959      0.564

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   250/3999      1.85G    0.04274   0.006141   0.001311          4        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.96it/s]
                   all         30         49      0.934      0.981      0.968      0.571

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   251/3999      1.85G    0.04414   0.007589   0.001693          7        320: 100% 8/8 [00:01<00:00,  6.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.07it/s]
                   all         30         49      0.896      0.917      0.922      0.548

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   252/3999      1.85G    0.04379   0.007172   0.001631          3        320: 100% 8/8 [00:01<00:00,  6.00it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.69it/s]
                   all         30         49      0.888      0.958      0.952      0.568

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   253/3999      1.85G    0.04294   0.007336   0.001718          7        320: 100% 8/8 [00:01<00:00,  6.02it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.78it/s]
                   all         30         49      0.921      0.917      0.935      0.564

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   254/3999      1.85G    0.04468    0.00778   0.001469          7        320: 100% 8/8 [00:01<00:00,  6.00it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.86it/s]
                   all         30         49      0.953      0.933      0.967      0.572

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   255/3999      1.85G    0.04263   0.007866   0.001535          7        320: 100% 8/8 [00:01<00:00,  6.31it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.07it/s]
                   all         30         49      0.978      0.933      0.969      0.566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   256/3999      1.85G    0.04451    0.00752   0.001578          6        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.13it/s]
                   all         30         49      0.939       0.94      0.958      0.546

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   257/3999      1.85G    0.04555   0.006291   0.001361          2        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.94it/s]
                   all         30         49      0.905      0.967      0.969      0.567

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   258/3999      1.85G    0.04149   0.007236   0.001377          5        320: 100% 8/8 [00:01<00:00,  6.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.37it/s]
                   all         30         49      0.919      0.963      0.955      0.573

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   259/3999      1.85G    0.04447   0.007749   0.001568          5        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.24it/s]
                   all         30         49      0.907      0.921      0.942      0.565

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   260/3999      1.85G    0.03994   0.006575   0.001563          4        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.14it/s]
                   all         30         49       0.93      0.967      0.964      0.534

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   261/3999      1.85G    0.04299   0.006922   0.001532          5        320: 100% 8/8 [00:01<00:00,  6.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.15it/s]
                   all         30         49      0.909       0.95      0.937      0.513

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   262/3999      1.85G    0.04165   0.007818   0.002142          9        320: 100% 8/8 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.24it/s]
                   all         30         49      0.885        0.9      0.904      0.532

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   263/3999      1.85G    0.04275   0.007862   0.001734          7        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.27it/s]
                   all         30         49      0.958       0.95      0.977      0.567

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   264/3999      1.85G    0.04317   0.006907   0.001363          4        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.44it/s]
                   all         30         49       0.97       0.94      0.972      0.584

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   265/3999      1.85G    0.04577   0.008575   0.001901         11        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.19it/s]
                   all         30         49      0.981      0.932      0.963       0.56

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   266/3999      1.85G    0.04198   0.007039   0.001446          4        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.87it/s]
                   all         30         49      0.933      0.945       0.96      0.558

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   267/3999      1.85G    0.04797   0.007668   0.001848          4        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.24it/s]
                   all         30         49      0.878      0.942       0.95      0.553

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   268/3999      1.85G      0.044   0.007464   0.002127          5        320: 100% 8/8 [00:01<00:00,  6.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.99it/s]
                   all         30         49      0.987      0.902      0.958      0.579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   269/3999      1.85G    0.04593   0.006918   0.001502          6        320: 100% 8/8 [00:01<00:00,  6.28it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.86it/s]
                   all         30         49      0.922      0.933      0.955      0.571

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   270/3999      1.85G     0.0447   0.007443   0.001402          7        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.15it/s]
                   all         30         49      0.917      0.933      0.946      0.575

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   271/3999      1.85G    0.04103   0.007176   0.001394          3        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.82it/s]
                   all         30         49      0.917      0.982      0.968      0.567

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   272/3999      1.85G    0.04258   0.006916   0.001401          3        320: 100% 8/8 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.79it/s]
                   all         30         49      0.899       0.95      0.946      0.564

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   273/3999      1.85G    0.04448   0.007224   0.001522          6        320: 100% 8/8 [00:02<00:00,  3.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.45it/s]
                   all         30         49       0.92      0.967      0.955      0.528

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   274/3999      1.85G    0.04377   0.007487   0.001596          5        320: 100% 8/8 [00:01<00:00,  4.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.81it/s]
                   all         30         49      0.904       0.95      0.952      0.537

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   275/3999      1.85G    0.04334   0.007716   0.001455          8        320: 100% 8/8 [00:01<00:00,  6.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.98it/s]
                   all         30         49       0.91      0.929      0.942      0.553

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   276/3999      1.85G    0.04472    0.00694   0.001416          5        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.61it/s]
                   all         30         49       0.92      0.943      0.945      0.552

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   277/3999      1.85G    0.04141   0.007348   0.001463          9        320: 100% 8/8 [00:01<00:00,  6.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.66it/s]
                   all         30         49      0.897       0.95      0.933      0.538

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   278/3999      1.85G    0.04552   0.007954   0.002339         11        320: 100% 8/8 [00:01<00:00,  6.07it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.09it/s]
                   all         30         49      0.908      0.917      0.936      0.518

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   279/3999      1.85G     0.0456   0.007039   0.001817          4        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.10it/s]
                   all         30         49      0.902       0.95       0.94      0.556

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   280/3999      1.85G    0.04214   0.007456   0.001443          8        320: 100% 8/8 [00:01<00:00,  6.32it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.00it/s]
                   all         30         49      0.964      0.945      0.966      0.553

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   281/3999      1.85G    0.04413   0.007143     0.0017          4        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.86it/s]
                   all         30         49       0.91       0.95      0.939       0.52

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   282/3999      1.85G    0.04342     0.0067    0.00161          3        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.85it/s]
                   all         30         49      0.904       0.95      0.945      0.531

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   283/3999      1.85G    0.04124   0.007096   0.001337          4        320: 100% 8/8 [00:02<00:00,  3.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.03it/s]
                   all         30         49      0.936      0.963      0.955      0.558

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   284/3999      1.85G    0.04159    0.00654   0.001283          5        320: 100% 8/8 [00:02<00:00,  3.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.69it/s]
                   all         30         49      0.932      0.958      0.975       0.55

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   285/3999      1.85G    0.04492   0.007169   0.001521          8        320: 100% 8/8 [00:01<00:00,  6.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.14it/s]
                   all         30         49      0.941       0.95      0.971       0.54

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   286/3999      1.85G    0.04268   0.006857   0.001271          6        320: 100% 8/8 [00:01<00:00,  5.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.13it/s]
                   all         30         49      0.937       0.98      0.971      0.542

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   287/3999      1.85G    0.04162   0.007573   0.001358          5        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.58it/s]
                   all         30         49      0.926      0.966      0.952      0.556

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   288/3999      1.85G    0.04148   0.007601   0.001379         11        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.64it/s]
                   all         30         49      0.903      0.967      0.948      0.556

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   289/3999      1.85G     0.0432   0.007696   0.001529          8        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.04it/s]
                   all         30         49      0.932      0.967      0.955      0.585

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   290/3999      1.85G    0.04444   0.008395   0.001504         14        320: 100% 8/8 [00:01<00:00,  6.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.96it/s]
                   all         30         49      0.895       0.95      0.942      0.576

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   291/3999      1.85G    0.04265   0.007734   0.001639          7        320: 100% 8/8 [00:01<00:00,  6.06it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.46it/s]
                   all         30         49      0.956      0.901      0.949      0.568

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   292/3999      1.85G    0.04295   0.007265     0.0013          9        320: 100% 8/8 [00:01<00:00,  6.06it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.97it/s]
                   all         30         49      0.959      0.915      0.942      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   293/3999      1.85G    0.04291   0.007273   0.001231          6        320: 100% 8/8 [00:01<00:00,  6.05it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.19it/s]
                   all         30         49       0.93      0.912      0.943      0.576

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   294/3999      1.85G     0.0417   0.007086   0.001621          6        320: 100% 8/8 [00:01<00:00,  6.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.70it/s]
                   all         30         49      0.939      0.934      0.961      0.567

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   295/3999      1.85G    0.04076   0.007706   0.001445          8        320: 100% 8/8 [00:01<00:00,  6.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.77it/s]
                   all         30         49      0.939      0.965      0.955      0.587

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   296/3999      1.85G    0.04218   0.006634   0.001447          3        320: 100% 8/8 [00:01<00:00,  6.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.87it/s]
                   all         30         49       0.94      0.964      0.955      0.594

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   297/3999      1.85G    0.04444   0.007872   0.001184         10        320: 100% 8/8 [00:01<00:00,  6.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.64it/s]
                   all         30         49      0.961      0.967      0.974      0.579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   298/3999      1.85G    0.04335   0.007112    0.00126          7        320: 100% 8/8 [00:01<00:00,  6.41it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.55it/s]
                   all         30         49      0.942      0.954      0.976      0.611

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   299/3999      1.85G    0.04247   0.007194   0.001461          5        320: 100% 8/8 [00:01<00:00,  6.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.86it/s]
                   all         30         49      0.951       0.97      0.979      0.596

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   300/3999      1.85G    0.04255   0.007873   0.001507         10        320: 100% 8/8 [00:01<00:00,  6.14it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.33it/s]
                   all         30         49      0.929       0.95      0.964      0.579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   301/3999      1.85G    0.04208   0.007624   0.001521          8        320: 100% 8/8 [00:01<00:00,  5.99it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.39it/s]
                   all         30         49      0.926       0.95      0.962      0.589

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   302/3999      1.85G    0.04141   0.007179   0.001563          4        320: 100% 8/8 [00:01<00:00,  6.07it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.35it/s]
                   all         30         49      0.954      0.967      0.977      0.595

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   303/3999      1.85G     0.0403    0.00653   0.001219          1        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.56it/s]
                   all         30         49       0.93      0.967      0.958      0.584

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   304/3999      1.85G    0.04073   0.006416   0.001496          4        320: 100% 8/8 [00:01<00:00,  6.07it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  6.04it/s]
                   all         30         49      0.895      0.967      0.954      0.564

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   305/3999      1.85G    0.04008   0.006782   0.001302          5        320: 100% 8/8 [00:01<00:00,  5.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.69it/s]
                   all         30         49      0.948      0.933      0.976      0.597

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   306/3999      1.85G    0.04298   0.007899    0.00149         12        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.90it/s]
                   all         30         49      0.922      0.967      0.955      0.564

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   307/3999      1.85G    0.04209    0.00726   0.001397          8        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.21it/s]
                   all         30         49      0.969      0.917      0.957      0.578

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   308/3999      1.85G    0.04097   0.006625   0.001314          2        320: 100% 8/8 [00:02<00:00,  3.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.57it/s]
                   all         30         49      0.931      0.983      0.975      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   309/3999      1.85G    0.04179   0.006837   0.001433          4        320: 100% 8/8 [00:02<00:00,  3.39it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.47it/s]
                   all         30         49      0.938      0.983      0.977      0.584

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   310/3999      1.85G    0.04416   0.006973   0.002165          5        320: 100% 8/8 [00:02<00:00,  3.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.01it/s]
                   all         30         49       0.93      0.967      0.963      0.601

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   311/3999      1.85G    0.03851   0.006395   0.001419          2        320: 100% 8/8 [00:01<00:00,  5.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.20it/s]
                   all         30         49      0.882       0.92      0.935      0.577

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   312/3999      1.85G    0.04138   0.006568   0.001481          4        320: 100% 8/8 [00:01<00:00,  6.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.88it/s]
                   all         30         49      0.929      0.967      0.967       0.57

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   313/3999      1.85G    0.04062   0.006879   0.001624          8        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.90it/s]
                   all         30         49      0.912      0.967      0.967      0.575

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   314/3999      1.85G    0.03969   0.006622   0.001301          4        320: 100% 8/8 [00:01<00:00,  6.30it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.67it/s]
                   all         30         49      0.929      0.967      0.958      0.551

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   315/3999      1.85G      0.042   0.006784   0.001605          7        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.07it/s]
                   all         30         49       0.94      0.967      0.962      0.561

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   316/3999      1.85G    0.04105   0.006823   0.001421          3        320: 100% 8/8 [00:01<00:00,  5.99it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.21it/s]
                   all         30         49      0.952      0.976       0.98      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   317/3999      1.85G    0.04205   0.006948   0.001604          6        320: 100% 8/8 [00:01<00:00,  5.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.51it/s]
                   all         30         49      0.964      0.933      0.973      0.554

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   318/3999      1.85G    0.04055   0.006845   0.001316          5        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.16it/s]
                   all         30         49      0.944      0.958      0.967      0.594

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   319/3999      1.85G    0.04293   0.006942   0.001304          7        320: 100% 8/8 [00:01<00:00,  6.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.93it/s]
                   all         30         49      0.942      0.956       0.96      0.598

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   320/3999      1.85G    0.04054   0.006471   0.001441          2        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.33it/s]
                   all         30         49      0.935      0.952      0.958      0.568

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   321/3999      1.85G    0.04252   0.006814   0.001592          6        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.59it/s]
                   all         30         49      0.961      0.967      0.978       0.59

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   322/3999      1.85G    0.04171   0.007216   0.001991          7        320: 100% 8/8 [00:01<00:00,  6.35it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.64it/s]
                   all         30         49      0.948      0.967      0.977      0.571

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   323/3999      1.85G     0.0412   0.007381   0.001429          6        320: 100% 8/8 [00:01<00:00,  6.01it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.34it/s]
                   all         30         49      0.943      0.962      0.959      0.563

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   324/3999      1.85G    0.04022   0.007162   0.001435          7        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.13it/s]
                   all         30         49      0.955      0.905      0.941      0.576

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   325/3999      1.85G    0.04396    0.00714   0.001186          6        320: 100% 8/8 [00:01<00:00,  5.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.75it/s]
                   all         30         49      0.943      0.948       0.97      0.574

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   326/3999      1.85G    0.03947     0.0068   0.001516          4        320: 100% 8/8 [00:01<00:00,  5.92it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.92it/s]
                   all         30         49      0.955      0.967       0.98      0.575

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   327/3999      1.85G    0.04112   0.007396   0.001328          8        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.92it/s]
                   all         30         49      0.947       0.95      0.952      0.571

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   328/3999      1.85G    0.04195   0.007507   0.001472         11        320: 100% 8/8 [00:01<00:00,  6.24it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.26it/s]
                   all         30         49      0.963      0.983       0.98      0.553

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   329/3999      1.85G    0.04133   0.006739   0.001393          6        320: 100% 8/8 [00:01<00:00,  6.10it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.99it/s]
                   all         30         49      0.963      0.983       0.98      0.583

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   330/3999      1.85G    0.04101   0.007245   0.001374          5        320: 100% 8/8 [00:01<00:00,  5.96it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.36it/s]
                   all         30         49      0.945      0.967       0.96      0.561

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   331/3999      1.85G    0.04201   0.007165   0.001315          6        320: 100% 8/8 [00:01<00:00,  6.00it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.53it/s]
                   all         30         49      0.929      0.967      0.957      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   332/3999      1.85G    0.04167   0.007953   0.001417          8        320: 100% 8/8 [00:01<00:00,  5.83it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.41it/s]
                   all         30         49      0.909       0.95      0.949       0.59

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   333/3999      1.85G    0.03866    0.00697   0.001386          3        320: 100% 8/8 [00:01<00:00,  5.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.74it/s]
                   all         30         49      0.901       0.95      0.945      0.596

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   334/3999      1.85G    0.04198   0.007317   0.001288          5        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.73it/s]
                   all         30         49      0.912      0.967       0.96      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   335/3999      1.85G    0.04685   0.006819   0.001941          2        320: 100% 8/8 [00:01<00:00,  5.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.14it/s]
                   all         30         49      0.961      0.917      0.951      0.594

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   336/3999      1.85G     0.0397   0.007245   0.002016          7        320: 100% 8/8 [00:01<00:00,  6.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.32it/s]
                   all         30         49      0.922      0.967      0.957      0.579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   337/3999      1.85G    0.04212   0.007374   0.001511          5        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.04it/s]
                   all         30         49      0.908       0.95      0.939       0.57

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   338/3999      1.85G     0.0418   0.006778   0.001564          5        320: 100% 8/8 [00:01<00:00,  6.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.16it/s]
                   all         30         49      0.918       0.95       0.95      0.524

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   339/3999      1.85G    0.04168   0.006111   0.001264          3        320: 100% 8/8 [00:01<00:00,  6.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.65it/s]
                   all         30         49      0.895      0.933      0.934      0.577

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   340/3999      1.85G    0.04045   0.007968   0.001494         12        320: 100% 8/8 [00:01<00:00,  5.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.84it/s]
                   all         30         49      0.981        0.9      0.952      0.553

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   341/3999      1.85G    0.04141   0.006901   0.001266          3        320: 100% 8/8 [00:01<00:00,  6.04it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.13it/s]
                   all         30         49      0.916      0.948      0.947      0.572

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   342/3999      1.85G    0.03987   0.006901   0.001143          3        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.29it/s]
                   all         30         49      0.935      0.954      0.969      0.578

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   343/3999      1.85G     0.0413   0.007839   0.001163         12        320: 100% 8/8 [00:01<00:00,  6.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.95it/s]
                   all         30         49      0.957      0.967       0.98      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   344/3999      1.85G    0.04081   0.007939   0.001249          9        320: 100% 8/8 [00:01<00:00,  5.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.75it/s]
                   all         30         49      0.954      0.967      0.979      0.604

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   345/3999      1.85G      0.042   0.006354   0.001413          3        320: 100% 8/8 [00:01<00:00,  6.22it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.05it/s]
                   all         30         49      0.973      0.967      0.977      0.579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   346/3999      1.85G    0.04346   0.007143   0.001329          3        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.63it/s]
                   all         30         49      0.941      0.983      0.972      0.569

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   347/3999      1.85G    0.04039   0.007827   0.001098         10        320: 100% 8/8 [00:01<00:00,  6.08it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.87it/s]
                   all         30         49      0.918      0.967      0.958      0.577

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   348/3999      1.85G    0.04052   0.006995   0.001208          7        320: 100% 8/8 [00:01<00:00,  6.06it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.76it/s]
                   all         30         49      0.933      0.967      0.956      0.572

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   349/3999      1.85G     0.0404   0.007071   0.001253          5        320: 100% 8/8 [00:01<00:00,  4.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.64it/s]
                   all         30         49      0.934      0.956      0.957      0.571

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   350/3999      1.85G    0.04135   0.007181   0.001432         10        320: 100% 8/8 [00:02<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.94it/s]
                   all         30         49      0.942      0.961      0.956      0.564

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   351/3999      1.85G    0.04115   0.007512   0.001343         10        320: 100% 8/8 [00:01<00:00,  4.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.85it/s]
                   all         30         49      0.936      0.967      0.959      0.583

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   352/3999      1.85G    0.04099   0.007625   0.001205          7        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.32it/s]
                   all         30         49      0.936      0.963      0.958      0.593

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   353/3999      1.85G    0.04005   0.006993   0.001511          4        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.99it/s]
                   all         30         49      0.935      0.953      0.958      0.589

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   354/3999      1.85G     0.0447   0.006977   0.001658          6        320: 100% 8/8 [00:01<00:00,  6.45it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.20it/s]
                   all         30         49      0.934      0.955      0.968      0.581

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   355/3999      1.85G    0.03972   0.007299   0.001701          6        320: 100% 8/8 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.18it/s]
                   all         30         49      0.919      0.967      0.968      0.572

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   356/3999      1.85G    0.04171   0.007282   0.001315         11        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.05it/s]
                   all         30         49      0.935      0.967      0.957       0.56

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   357/3999      1.85G    0.03838   0.007171   0.001204          6        320: 100% 8/8 [00:01<00:00,  6.23it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.23it/s]
                   all         30         49      0.934      0.967      0.957      0.598

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   358/3999      1.85G    0.04104   0.007309   0.001303          9        320: 100% 8/8 [00:01<00:00,  6.05it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.19it/s]
                   all         30         49      0.939      0.921       0.95      0.581

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   359/3999      1.85G    0.03983   0.006874   0.001225          7        320: 100% 8/8 [00:01<00:00,  5.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.22it/s]
                   all         30         49      0.935      0.962      0.959      0.606

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   360/3999      1.85G     0.0399   0.007266     0.0012          5        320: 100% 8/8 [00:02<00:00,  3.85it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.99it/s]
                   all         30         49      0.929      0.963      0.957       0.59

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   361/3999      1.85G    0.03953   0.006713   0.001165          4        320: 100% 8/8 [00:01<00:00,  4.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.24it/s]
                   all         30         49      0.915      0.967      0.954      0.607

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   362/3999      1.85G    0.04053   0.006294   0.001514          5        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.12it/s]
                   all         30         49      0.917      0.965      0.958      0.597

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   363/3999      1.85G    0.04047   0.007233   0.001565          7        320: 100% 8/8 [00:01<00:00,  6.00it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.79it/s]
                   all         30         49      0.935      0.964      0.959      0.586

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   364/3999      1.85G    0.04058   0.007253   0.001353          7        320: 100% 8/8 [00:01<00:00,  5.99it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.58it/s]
                   all         30         49      0.956       0.98      0.976      0.599

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   365/3999      1.85G    0.04088   0.007061    0.00141          7        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.94it/s]
                   all         30         49      0.941      0.964      0.957      0.587

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   366/3999      1.85G    0.04025   0.006418   0.001441          3        320: 100% 8/8 [00:01<00:00,  5.96it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.31it/s]
                   all         30         49       0.94      0.959      0.959      0.603

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   367/3999      1.85G    0.03992   0.007575   0.001419          8        320: 100% 8/8 [00:01<00:00,  5.93it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.13it/s]
                   all         30         49      0.953      0.983      0.975      0.559

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   368/3999      1.85G    0.04016   0.007476   0.001274          8        320: 100% 8/8 [00:01<00:00,  6.02it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.68it/s]
                   all         30         49      0.962      0.983      0.975      0.581

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   369/3999      1.85G    0.04012   0.006883   0.001218          8        320: 100% 8/8 [00:01<00:00,  6.17it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.78it/s]
                   all         30         49      0.961      0.983      0.975      0.583

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   370/3999      1.85G    0.03862   0.006645   0.001272          6        320: 100% 8/8 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.70it/s]
                   all         30         49      0.955      0.983      0.976      0.578

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   371/3999      1.85G    0.03929   0.006101   0.001316          3        320: 100% 8/8 [00:01<00:00,  6.19it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.77it/s]
                   all         30         49      0.932      0.967      0.966      0.565

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   372/3999      1.85G    0.03856    0.00754   0.001249          6        320: 100% 8/8 [00:01<00:00,  6.18it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.00it/s]
                   all         30         49       0.92      0.967      0.968      0.581

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   373/3999      1.85G    0.03971   0.006359   0.001564          3        320: 100% 8/8 [00:01<00:00,  6.04it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.93it/s]
                   all         30         49      0.907      0.967      0.951      0.578

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   374/3999      1.85G      0.042   0.006765   0.001344          7        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.79it/s]
                   all         30         49      0.936      0.967      0.955      0.563

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   375/3999      1.85G    0.03775   0.006725    0.00128          4        320: 100% 8/8 [00:01<00:00,  6.07it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.24it/s]
                   all         30         49      0.941      0.983      0.972      0.579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   376/3999      1.85G    0.03577   0.005621     0.0014          0        320: 100% 8/8 [00:01<00:00,  6.11it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.07it/s]
                   all         30         49      0.942      0.967      0.956      0.565

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   377/3999      1.85G    0.03963   0.006899   0.001326          3        320: 100% 8/8 [00:01<00:00,  5.92it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.64it/s]
                   all         30         49      0.929      0.967      0.957      0.558

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   378/3999      1.85G    0.04126   0.006983   0.001601          5        320: 100% 8/8 [00:01<00:00,  6.12it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.79it/s]
                   all         30         49      0.928      0.967      0.957      0.561

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   379/3999      1.85G    0.04473   0.007648    0.00162          7        320: 100% 8/8 [00:01<00:00,  6.21it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.00it/s]
                   all         30         49      0.933      0.967      0.967      0.557

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   380/3999      1.85G    0.04032   0.006609   0.001337          2        320: 100% 8/8 [00:01<00:00,  6.26it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.10it/s]
                   all         30         49      0.929      0.967      0.969      0.573

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   381/3999      1.85G    0.03775   0.007159   0.001261          5        320: 100% 8/8 [00:01<00:00,  6.16it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.58it/s]
                   all         30         49      0.915       0.95      0.937      0.563

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   382/3999      1.85G    0.04086    0.00676   0.001202          4        320: 100% 8/8 [00:01<00:00,  5.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.80it/s]
                   all         30         49      0.952      0.983      0.977      0.559

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   383/3999      1.85G    0.03755   0.006143   0.001525          2        320: 100% 8/8 [00:01<00:00,  6.37it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.54it/s]
                   all         30         49      0.917       0.95      0.945      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   384/3999      1.85G    0.03992   0.007772   0.001282         14        320: 100% 8/8 [00:01<00:00,  6.34it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.85it/s]
                   all         30         49      0.938      0.967      0.958      0.579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   385/3999      1.85G    0.04063   0.007303   0.001301          8        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.28it/s]
                   all         30         49      0.956      0.983      0.978      0.578

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   386/3999      1.85G    0.03953   0.006328    0.00114          7        320: 100% 8/8 [00:01<00:00,  5.83it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.66it/s]
                   all         30         49      0.935      0.983      0.978      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   387/3999      1.85G    0.03742   0.007124   0.001265          3        320: 100% 8/8 [00:01<00:00,  5.95it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.49it/s]
                   all         30         49      0.933      0.944      0.964      0.594

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   388/3999      1.85G    0.03991   0.007155   0.001215          4        320: 100% 8/8 [00:01<00:00,  5.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.07it/s]
                   all         30         49      0.928      0.961      0.967      0.579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   389/3999      1.85G    0.03881   0.007774   0.001154         11        320: 100% 8/8 [00:01<00:00,  6.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.56it/s]
                   all         30         49      0.951      0.979      0.976      0.593

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   390/3999      1.85G    0.04088   0.006868   0.001107          8        320: 100% 8/8 [00:01<00:00,  6.27it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.99it/s]
                   all         30         49       0.92      0.933      0.937       0.58

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   391/3999      1.85G    0.03718   0.006347   0.001268          4        320: 100% 8/8 [00:01<00:00,  6.04it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.88it/s]
                   all         30         49      0.923      0.939      0.939      0.567

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   392/3999      1.85G    0.04133   0.007677   0.001222          9        320: 100% 8/8 [00:01<00:00,  6.33it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.92it/s]
                   all         30         49       0.95      0.978      0.978      0.587

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   393/3999      1.85G    0.04158   0.006774   0.001197          5        320: 100% 8/8 [00:01<00:00,  5.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.95it/s]
                   all         30         49      0.893      0.919      0.916      0.559

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   394/3999      1.85G    0.04134   0.007205   0.001417          8        320: 100% 8/8 [00:01<00:00,  6.09it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.12it/s]
                   all         30         49      0.922      0.951      0.942      0.566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   395/3999      1.85G    0.03718   0.006724   0.001077          3        320: 100% 8/8 [00:01<00:00,  5.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  5.61it/s]
                   all         30         49      0.932       0.95      0.959      0.553

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   396/3999      1.85G    0.03974   0.006968   0.001271          6        320: 100% 8/8 [00:01<00:00,  5.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.93it/s]
                   all         30         49      0.942       0.95      0.968      0.546

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   397/3999      1.85G    0.03938   0.006168   0.001252          3        320: 100% 8/8 [00:01<00:00,  6.20it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.62it/s]
                   all         30         49       0.94      0.937      0.967      0.582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
   398/3999      1.85G    0.03761   0.006081    0.00125          2        320: 100% 8/8 [00:01<00:00,  5.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.95it/s]
                   all         30         49      0.943      0.933      0.956      0.597
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 298, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

399 epochs completed in 0.214 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.3MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model summary: 157 layers, 7020913 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.73it/s]
                   all         30         49      0.942      0.954      0.976       0.61
                    40         30         10      0.967          1      0.995      0.765
                    45         30         16      0.998          1      0.995       0.56
                    50         30         15      0.924      0.815       0.92      0.422
                    65         30          8      0.878          1      0.995      0.694
Results saved to runs/train/exp
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:      metrics/mAP_0.5 ▁▃▅▆▇▇▇▇████████████████████████████████
wandb: metrics/mAP_0.5:0.95 ▁▂▃▄▅▅▆▆▇▆▆▇▇▆▇▇▇▆▇▇▇▇██▇▇▇▇▇██▇████▇███
wandb:    metrics/precision ▁▆▇▅▇▇▇▇█▇▆█▇▇▇▇▇▇█▇▇█▇█▇█▇█████████████
wandb:       metrics/recall ▁▃▄▅▆▇▇▇█▇██▇▇█▇█▇▇██▇█▇██████▇██▇██████
wandb:       train/box_loss █▇▄▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/cls_loss █▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/obj_loss ▄▅▇█▆▆▅▅▃▅▄▃▅▄▄▄▃▄▄▂▃▃▃▂▃▃▄▃▃▄▂▁▃▃▄▃▃▁▁▂
wandb:         val/box_loss █▇▅▄▃▃▂▂▂▂▂▂▂▂▂▂▁▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         val/cls_loss █▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         val/obj_loss ▃▂█▆▅▄▄▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                x/lr0 █▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                x/lr1 ▁█████████████████████▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
wandb:                x/lr2 ▁█████████████████████▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
wandb: 
wandb: Run summary:
wandb:           best/epoch 298
wandb:         best/mAP_0.5 0.97633
wandb:    best/mAP_0.5:0.95 0.61058
wandb:       best/precision 0.94189
wandb:          best/recall 0.95373
wandb:      metrics/mAP_0.5 0.97633
wandb: metrics/mAP_0.5:0.95 0.61024
wandb:    metrics/precision 0.94191
wandb:       metrics/recall 0.95373
wandb:       train/box_loss 0.03761
wandb:       train/cls_loss 0.00125
wandb:       train/obj_loss 0.00608
wandb:         val/box_loss 0.03329
wandb:         val/cls_loss 0.00115
wandb:         val/obj_loss 0.00485
wandb:                x/lr0 0.00902
wandb:                x/lr1 0.00902
wandb:                x/lr2 0.00902
wandb: 
wandb: Synced devoted-lion-4: https://wandb.ai/jgba/YOLOv5/runs/i2cnnlrc
wandb: Synced 5 W&B file(s), 13 media file(s), 1 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20221103_231305-i2cnnlrc/logs

  ```
</details>

### Evidências do treinamento

O modelo visa localizar 4 karts ao longo de retratos retirados de filmagens (nºs 40, 45, 50 e 65)

Matriz de confusão
![Matriz de confusão](https://user-images.githubusercontent.com/116410211/199860999-8900bd40-1baa-4be0-b564-e62f03df778b.png)

Métricas
![Metricas](https://user-images.githubusercontent.com/116410211/199861153-44cecb84-4489-4cf7-b77e-677fc9cdef62.png)

![Metricas](https://user-images.githubusercontent.com/116410211/199859458-9be294df-d880-4713-9f03-243758b99ea5.png)

IA localizando placas
![IA localizando placas](https://user-images.githubusercontent.com/116410211/199861282-0ee29c43-d800-4d01-8672-a6dd2b386fe4.png)

![IA localizando placas](https://user-images.githubusercontent.com/116410211/199859542-2e466fb5-3627-4958-9187-8c425ffa6eee.png)

![IA localizando placas](https://user-images.githubusercontent.com/116410211/199859595-78da0210-3c4a-4855-ab3d-328798057201.png)




## Roboflow

[kart_plates](https://universe.roboflow.com/kartplates/kart_plates)

## HuggingFace

Nessa seção você deve publicar o link para o HuggingFace
