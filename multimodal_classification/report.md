# Assignment 1: Deep Learning Classification Suite

Multimodal: Zero-shot vs. Few-shot\
Author: Nguyen Viet Hoang

## Abstract
Apply the **CLIP** model with classification head added to accurately classify image-text pairs belong to a class. Compare zero-shot performance against few-shot fine-tuning.

## Implementaion
### Input
Let's load dataset.
```python
dataset = load_dataset("ashraq/fashion-product-images-small", split="train", cache_dir='/kaggle/working/dataset_cache',streaming=False)
dataset = dataset.select_columns(["image", "productDisplayName", 'subCategory'])
dataset
```
```python
Dataset({
    features: ['image', 'productDisplayName', 'subCategory'],
    num_rows: 44072
})
```
Next, check validity of the data. Fortunately, this dataset is already valid! (for all runs by author)
```python
for sample in dataset:
  if not (sample['image'] and sample['productDisplayName'] and sample['subCategory']): raise ValueError
```
Let's see dataset labels:
```python
from collections import Counter
cls_counter = Counter(dataset["subCategory"])
print(len(cls_counter))
print(cls_counter)
```
```python
45
Counter({'Topwear': 15383, 'Shoes': 7323, 'Bags': 3053, 'Bottomwear': 2685, 'Watches': 2542, 'Innerwear': 1806, 'Jewellery': 1079, 'Eyewear': 1073, 'Fragrance': 1001, 'Sandal': 961, 'Wallets': 925, 'Flip Flops': 913, 'Belts': 811, 'Socks': 698, 'Dress': 478, 'Loungewear and Nightwear': 464, 'Saree': 427, 'Lips': 425, 'Headwear': 293, 'Nails': 278, 'Makeup': 263, 'Ties': 258, 'Accessories': 129, 'Scarves': 118, 'Cufflinks': 108, 'Apparel Set': 106, 'Free Gifts': 104, 'Stoles': 90, 'Skin': 53, 'Skin Care': 49, 'Mufflers': 38, 'Eyes': 34, 'Sports Equipment': 21, 'Gloves': 20, 'Hair': 19, 'Bath and Body': 9, 'Water Bottle': 7, 'Perfumes': 6, 'Umbrellas': 6, 'Shoe Accessories': 4, 'Wristbands': 4, 'Beauty Accessories': 3, 'Sports Accessories': 3, 'Home Furnishing': 1, 'Vouchers': 1})
```
The classes are pretty imbalanced. Let's keep it in mind in case it causes problems later on.

Next, let's see some data samples. See here for some samples: https://colab.research.google.com/drive/1MZ20ccSe01mQEzUmACtCrHNOD3TC0ga4#scrollTo=4HRY_zfgcs2E&line=1&uniqifier=1
```python
from IPython.display import display
for i in range(10):
    display(dataset[i]['image'])
    print(f'{dataset[i]['productDisplayName'], dataset[i]['subCategory']}')
```
Product names are pretty length varied. Shorter names can be padded to some fixed predefined max length.

Next, let's see image size:
```python
dataset[:10]['image']
```
```python
[<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=60x80>,
 <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=60x80>,
 <PIL.Image.Image image mode=L size=60x80>,
 <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=60x80>,
 <PIL.Image.Image image mode=RGB size=60x80>,
 <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=60x80>,
 <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=60x80>,
 <PIL.Image.Image image mode=RGB size=60x80>,
 <PIL.Image.Image image mode=RGB size=60x80>,
 <PIL.Image.Image image mode=RGB size=60x80>]
```
These are coloured images, so the size is (3,60,80), which is not too large. However, as Huggingface's pretrained CLIP model will be used, it requires that image input size be (3, 224, 224), which is larger than expected. Fortunately, it's a much smaller cost in time and effort to use pretrained models than to train from scratch.

#### Dataloader preprocessing
To train, it requires batch dataloader.

First, rename columns to more generic ones:
```python
ds = dataset.cast_column('subCategory', ClassLabel(names=dataset.unique('subCategory')))
ds = ds.rename_columns({
    "productDisplayName": "text",
    "subCategory": "label"
})
ds
```
```python
Dataset({
    features: ['image', 'text', 'label'],
    num_rows: 44072
})
```
Let's partition data to train, val, and test set.
```python3
import torch
torch.manual_seed(42)
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]
print(len(train_ds), len(test_ds))
# (35257, 8815)

torch.manual_seed(42)
train_size = int(0.8*len(train_ds))
val_size = len(train_ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
print(len(train_ds), len(val_ds))
# (28205, 7052)
```
Before loading to Dataloader, as CLIP model will be used, it requires proper input format for both image and text. CLIP's built-in processor can be used to ease much effort for raw input image and text, but it requires collate_fn be defined for custom batch processing, rather than dataloader's default tensor format in batch, which requires complex raw image and text preprocessing.

Let's use CLIP built-in models and processor.
```python
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection #CLIPModel
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32",output_attentions=True).to(device)
        self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32",output_attentions=True).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.parameters():
            param.requires_grad = False

    def collate_fn_image_text_label(self, batch):
        images = [item['image'] for item in batch]
        text = [item['text'] for item in batch]
        with torch.no_grad():
            image_inputs = self.processor(images=images, return_tensors="pt").to(device)
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True).to(device)
            image_embeds = self.vision_model(**image_inputs).image_embeds
            text_embeds = self.text_model(**text_inputs).text_embeds
            X = torch.cat((image_embeds.unsqueeze(1), text_embeds.unsqueeze(1)), dim=1)
            y = torch.tensor([item['label'] for item in batch])
            return X, y

    def num_parameters(self):
        return self.vision_model.num_parameters() + self.text_model.num_parameters()
```
As can be seen, collate_fn* returns image and text embeds from CLIP pretrained models, which is easy and saves much effort. Then, it simply treats the two as features from dataset.

Finally, dataloader is used for train, val, test set.
```python
clip_model = FCLIP()
clip_model.num_parameters()
# 151277312
```
```python
torch.manual_seed(42)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, collate_fn=clip_model.collate_fn_image_text_label, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, collate_fn=clip_model.collate_fn_image_text_label, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, collate_fn=clip_model.collate_fn_image_text_label, shuffle=False)
```

### Model
Before defining the model, let's look at helping classes and functions.
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss, model, optimizer, scheduler):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": self.best_loss
            }, f"checkpoint_{model.name}.pt")
            return False  # don't stop
        else:
            self.counter += 1

            return self.counter >= self.patience

    def reset(self):
        self.best_loss = float('inf')
        self.counter = 0
```
```python3
from torchmetrics import Accuracy, F1Score
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TorchMetrics:
    def __init__(self, num_classes, acc=True, f1=True):
        self.metrics = {}
        if acc: self.metrics['accuracy'] = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        if f1: self.metrics['f1'] = F1Score(task="multiclass", num_classes=num_classes, average='weighted').to(device)

    def update(self, preds, labels):
        for i in self.metrics: self.metrics[i].update(preds, labels)

    def compute(self):
        return {k: v.compute().item() for k, v in self.metrics.items()}

    def reset(self):
        for i in self.metrics: self.metrics[i].reset()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(checkpoint_path, model, optimizer=None, scheduler=None, early_stopping=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if early_stopping: early_stopping.best_loss = checkpoint['val_loss']

def train_model(model, criterion, optimizer: torch.optim.Optimizer, train_loader, val_loader, scheduler=None, resume=False, num_epochs=50, seed=42):
    torch.manual_seed(seed)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    log_trainloss, log_valloss = [], []
    if resume:
        load_model(f"checkpoint_{model.name}.pt", model, optimizer, scheduler, early_stopping)
        for param_group in optimizer.param_groups: param_group['lr'] = 0.0001

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x ,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        if scheduler: scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output,y)
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            log_trainloss.append(train_loss)
            log_valloss.append(val_loss)
            print(f"Epoch {epoch+1}: loss {train_loss}, val_loss {val_loss}")

            if early_stopping.step(val_loss, model, optimizer, scheduler):
                print("Early Stopping")
                break
    return log_trainloss, log_valloss

def evaluate_model(model, test_loader, torch_metrics):
    model.eval()
    with torch.no_grad():
        torch_metrics.reset()
        # correct = 0
        # total = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            predicted = output.argmax(dim=1)
            torch_metrics.update(predicted,y)
        #     total += y.size(0)
        #     correct += (predicted == y).sum().item()

        # print(total, correct)
        # print(f"Accuracy: {100 * correct / total}%")
        print(torch_metrics.compute())
```
Above functions are relatively easy to understand, so let's focus on 2 functions below. They are created for few-shot classification task. The first one is to evaluate the default CLIP model with and without added model. Default CLIP operation in this function takes average of the 2 CLIP models' embeddings, while extended CLIP instead learns relative importance of the 2 embeddings. Then, the function performs similarity of combined embedding with each label's embedding, with output of the most similar label. The label embedding can be average of all samples' embeddings, each from either default CLIP average embedding of image and text or the model's learned combination of 2 embeddings.
```python
def few_shot(label_embeds, test_loader, torch_metrics, model=None, test_nums=None):
    with torch.no_grad():
        torch_metrics.reset()
        # correct = 0
        # total = 0
        if not test_nums: test_nums = len(test_loader)
        for (x, y) in (test_loader):
            x, y = x.to(device), y.to(device)
            image_embed, text_embed = x.unbind(dim=1)
            if model: output = model(x)
            else: output = (image_embed + text_embed)/2
            predicted = (output @ label_embeds.T).argmax(dim=1)
            torch_metrics.update(predicted,y)
            # total += y.size(0)
            # correct += (predicted == y).sum().item()
            # if i > test_nums: break
        # print(total, correct)
        # print(f"Accuracy: {100 * correct / total}%")
        print(torch_metrics.compute())

def embed_label(train_loader, num_class, default_projection_dim=512, model=None):
    with torch.no_grad():
        projection_dim = default_projection_dim if not model else model.projection_dim
        sum_embed = torch.zeros(num_class,projection_dim).to(device)
        count = torch.zeros(len(cls_counter), dtype=torch.long).to(device)
        for (X, y) in (train_loader):
            X, y = X.to(device), y.to(device)
            image_embed, text_embed = X.unbind(dim=1)
            if model: output = model(X)
            else: output = (image_embed + text_embed)/2
            sum_embed.index_add_(0, y, output)
            count.index_add_(0, y, torch.ones_like(y))
        return sum_embed / count.clamp(min=1).unsqueeze(-1)
```
It can be noted that there is no zero-shot function for reasons following. If the pretraining set can be considered as "few" shot for a model to learn to classify in test set, the model can be said to perform few-shot classification. As "few" can be relative, the definition for few-shot may be applied.
However, if few-shot classification is defined as not learning, or updating, parameters, then label embedding has to be explicitly defined from static model's output embedding of a few samples. As such, those non-updating, static, or frozen model that does not use few examples explicitly to classify may be considered zero-shot.

Let's perform zero-few shot classification for example. Label embed is from CLIP text embedding of text label "subCategory", which is pretrained with "few" examples. However, it does not explicitly define a few examples, so from non-updating model's perspective, it is a zero-shot classification.
```python
label_inputs = clip_model.processor(text=label, return_tensors="pt", padding=True).to(device)
label_embeds = clip_model.text_model(**label_inputs).text_embeds
label_embeds.shape
# torch.Size([45, 512])

few_shot(label_embeds,test_loader,torch_metrics) # few-shot ~ zero-shot, relative to CLIP pretraining set
# {'accuracy': 0.615995466709137, 'f1': 0.6612886190414429}
```
Look at the accuracy and f1 score. For a 150M-parameter model, it should have achieved much better than that. Let's try to extend the CLIP model.

To support CLIP models on the dataset, let's define simple head of attention over CLIP embeddings of image and text, and n free tokens to fine-tune embedding transformation.
```python
import torch.nn as nn
class Attention2(nn.Module):
    def __init__(self, embed_dim=512, nhead=8):
        super().__init__()
        self.nhead = nhead
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        B, N, D = x.shape
        head_dim = D//self.nhead
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.nhead, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = x / x.norm(dim=-1, keepdim=True) #(2)
        return x
    def forward_attn(self, x):
        B, N, D = x.shape
        head_dim = D//self.nhead
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.nhead, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        return attn

class FreeTransformer2(nn.Module):
    def __init__(self, num_token=3, embed_dim=512, nhead=8, num_class=10):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_class)
        self.n_token = nn.Parameter(torch.randn(1, num_token, embed_dim))
        self.attn = Attention2(embed_dim=embed_dim, nhead=nhead)
        self.name = "free_transformer"
        self.projection_dim = num_class
    def forward(self, x):
        B = x.shape[0]
        n_token = self.n_token.expand(B, -1, -1)
        num_token = n_token.shape[1]
        x = torch.cat((x, n_token), dim=1)
        x = self.attn(x)
        x = x[:, 0]
        x = self.head(x)
        # x = x / x.norm(dim=-1,keepdim=True) (!)
        return x
    def forward_attn(self, x):
        B = x.shape[0]
        n_token = self.n_token.expand(B, -1, -1)
        num_token = n_token.shape[1]
        x = torch.cat((x, n_token), dim=1)
        attn = self.attn.forward_attn(x)
        first_token_attn = attn[:, :, 0, :]
        return first_token_attn.mean(dim=1)
```
Class Attention2 defines a simple attention layer, with method forward_attn for later attention visualization.
Class FreeTransformer2 applies the attention layer over token-like features, and takes the first feature to a head of num_class; x is concat of image and text embeddings from CLIP models, which is also concat with n free tokens. The forward_attn method is for later inspecting which features are important, image or text.

Let's apply simple n=1 free token.
```python
from time import time
seed = int(time())
print('seed:',seed)
torch.manual_seed(seed)

embed_dim = clip_model.vision_model.config.projection_dim

free_model = FreeTransformer2(num_token=1, embed_dim=embed_dim, nhead=8, num_class=len(cls_counter))
free_model = free_model.to(device)

optimizer = torch.optim.AdamW(free_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
# try: load_model(f"checkpoint_{free_model.name}.pt", free_model, optimizer, scheduler)
# except: print("No checkpoint found")

print(free_model)
print(count_parameters(free_model))
```
```python
seed: 1775435470
FreeTransformer2(
  (head): Linear(in_features=512, out_features=45, bias=True)
  (attn): Attention2(
    (qkv): Linear(in_features=512, out_features=1536, bias=True)
    (proj): Linear(in_features=512, out_features=512, bias=True)
  )
)
1074221
```
It can be seen that even though there is only 1 token, the model is already quite large, with over 1M parameters most of which are from attention layer and its high dimensional features. However, looking back at CLIP number of parameters, it is much fortunately smaller and easier to train.

Let's train and see how well the model learns.
```python
log_trainloss, log_valloss = train_model(free_model, nn.CrossEntropyLoss(), optimizer, train_loader, val_loader, scheduler=scheduler, num_epochs=10)
```
```python
Epoch 1: loss 0.6912520158985237, val_loss 0.15076377987861633
Epoch 2: loss 0.1020660184855972, val_loss 0.09210804104804993
Epoch 3: loss 0.0620638840417895, val_loss 0.06971453875303268
Epoch 4: loss 0.04541044475831717, val_loss 0.06445742398500443
Epoch 5: loss 0.03486491310834682, val_loss 0.059799548238515854
Epoch 6: loss 0.029158952141491074, val_loss 0.057783570140600204
Epoch 7: loss 0.022279595538853433, val_loss 0.050061196088790894
Epoch 8: loss 0.017708191778399936, val_loss 0.05077046528458595
Epoch 9: loss 0.014678119860226267, val_loss 0.05037644878029823
Epoch 10: loss 0.012925738715587814, val_loss 0.05003775283694267
```
What a success! Even a very simple extended model like this can achieve very small loss. Let's plot this training progress.
```python
import matplotlib.pyplot as plt

# Convert log_valloss to a list of CPU numbers
log_valloss_cpu = [loss.cpu().item() for loss in log_valloss]

plt.plot(log_trainloss, label='Train')
plt.plot(log_valloss_cpu, label='Val')
plt.legend()
plt.show()
```
[Plot of training progress](img-link)

Let's try to evaluate on test set. This will be zero-shot classification.
```python
evaluate_model(free_model,test_loader,torch_metrics)
```
```python
{'accuracy': 0.9908111095428467, 'f1': 0.9897392988204956}
```
Great to see! Much higher accuracy and f1 score for very simple extended CLIP than its default model. Let's see how few-shot performs.
```python
few_shot(embed_label(train_loader,len(cls_counter),model=free_model),test_loader,torch_metrics,model=free_model)
```
```python
{'accuracy': 0.9804878234863281, 'f1': 0.9799184203147888}
```
Not as high as one might expect. However, not too surprising as simple average class embedding of all samples for the same label might not be as fine-tuned as that the network learns itself; just like a simple average of CLIP models is much worse than its learnable combination by the network, as can be shown in attention visualization.

### Visualize and interpret model
Let's write simple functions returning attention layer from CLIP models and extended one.
```python
import random
def clip_image_attn(clip_model,sample):
    image_inputs = clip_model.processor(images=sample['image'], return_tensors="pt").to(device)
    image_output = clip_model.vision_model(**image_inputs,output_attentions=True)
    last_layer_attn = image_output.attentions[-1]

    cls_attn = last_layer_attn[:, :, 0, 1:] # [batch, nheads, seq_len]
    avg_cls_attn = cls_attn.mean(dim=1) # [batch, seq_len]
    return avg_cls_attn

def clip_text_attn(clip_model,sample):
    text_inputs = clip_model.processor(text=sample['text'], return_tensors="pt", padding=True).to(device)
    text_output = clip_model.text_model(**text_inputs, output_attentions=True)
    last_layer_attn = text_output.attentions[-1]

    first_eos_idx = text_inputs['attention_mask'].sum(dim=1)-1
    # or use:
    # eos_token_id = clip_model.processor.tokenizer.eos_token_id
    # first_eos_idx = (text_inputs["input_ids"] == eos_token_id).int().argmax(dim=1)

    # Attention from the EOS token to all other tokens
    eos_attn = last_layer_attn[torch.arange(last_layer_attn.size(0)), :, first_eos_idx, :]  # [batch, nheads, seq_len]
    return eos_attn.mean(dim=1)

def free_model_attn(free_model,clip_model,sample):
    sample = [{'image':img, 'text':txt, 'label':lbl} for img, txt, lbl in zip(sample['image'], sample['text'], sample['label'])]
    X, _ = clip_model.collate_fn_image_text_label(sample)
    return free_model.forward_attn(X)

ridx = random.randint(0,len(test_ds)-10)
samples = test_ds[ridx : ridx+10]

with torch.no_grad():
  image_attn, free_attn, text_attn = clip_image_attn(clip_model,samples), free_model_attn(free_model,clip_model,samples), clip_text_attn(clip_model,samples)
  patch_importance = free_attn[:,0].unsqueeze(1) * image_attn
  word_importance = free_attn[:,1].unsqueeze(1) * text_attn
patch_importance.shape, word_importance.shape
# (torch.Size([10, 49]), torch.Size([10, 14]))
```
This results in shape of importance scores for image patches and text tokens. It can be seen that there are 49 patches inside an image and 14 tokens inside a text sentence.

Next, let's visualize which patch and token is important.
```python
import numpy as np
from scipy.ndimage import zoom

def plot_heatmap(images, patch_importance, h=60, w=80):
    heatmap = patch_importance.reshape(10, 7, 7).cpu().numpy()
    # Normalise for display
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_resized_batch = zoom(heatmap, (1, h/7, w/7), order=1)

    for i in range(len(images)):
        plt.figure(figsize=(5, 5))
        plt.imshow(images[i])
        plt.imshow(heatmap_resized_batch[i], cmap='hot', alpha=0.5)
        plt.axis('off')
        plt.title(f"Sample {i+1}")
        plt.show()
plot_heatmap(samples['image'], patch_importance)

from IPython.display import display, HTML
def displayHtml_token(processor, text, word_importance):
    text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    tokens = processor.tokenizer.convert_ids_to_tokens(text_inputs["input_ids"][0])
    # Normalise per word
    word_importance = (word_importance - word_importance.min()) / (word_importance.max() - word_importance.min() + 1e-8)

    html = "<div style='font-size:14px;'>"
    for token, w in zip(tokens, word_importance):
        if token in [processor.tokenizer.cls_token, processor.tokenizer.sep_token, processor.tokenizer.pad_token]: continue  # skip special tokens

        color = int(255 * (1 - w))
        html += f"<span style='background-color: rgb(255,{color},{color});'>{token} </span>"
    html += "</div>"
    display(HTML(html))

def displayHtml_batchtoken(processor, batch_text, word_importance):
    for i in range(len(batch_text)):
        print(f'sample {i}: {batch_text[i]}')
        displayHtml_token(processor, batch_text[i], word_importance[i])

displayHtml_batchtoken(clip_model.processor, samples['text'], word_importance)
```
See here are some results: https://colab.research.google.com/drive/1MZ20ccSe01mQEzUmACtCrHNOD3TC0ga4#scrollTo=EXzuFPkmUjB-&line=7&uniqifier=1

It can be seen that the attention layer is quite accurate in which region importance determines the outcomes.
Then, to see which feature image or text is more important:
```python
free_attn
```
It can be seen that text feature takes most of the importance, next image, and very insignificantly free token for the 10 test samples. Perhaps, we can try to see and refine in the future how more free tokens would allow better fine-tuning for the net. For now, it is the most weighted attention layer that is the culprit for high accuracy on this dataset.

## **Key Achievement**
With zero shot's 99% accuracy on the test set, few shot can barely improves it. In most of the runs few-shot is actually slightly less accurate, with some rare occasion of insignificant improvement.

---

## 🛠 Extensions (Bonus 40%)
* **Interpretability:** Integrated Attention Maps from regions of image and text to visualize model decisions.
* **Augmentation**: No augmentation needed for very high accuracy on this dataset.
* **Ensemble and fine-tuning**: Combine CLIP image and text projection model to a simple free head of attention. This results in very high accuracy compared to simple static average of 2 CLIP models; the network learns proper attention to image and text itself. The free 512-dimension n tokens may allow the model to freely fine-tune image and text feature transformation from CLIP models that leads to final classification.
