import torch

class Trans_config:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    typ = "max"
    head = "l3"
    loss = "aslcb"

    name = f"trans_{loss}{f"_{typ}" if typ and typ == "lwa" else ""}_{head}_best.pt"
    path = f'model/weight/eurlex/{name}'
    epochs=30
    
    thres= 0.7


"""
    
trans_aslcb = train_transformer(trans_aslcb, train_loader, test_loader, mlb, dev, 
                          epochs=40, 
                          head_name='l3', # Flat label no hierarchy 
                          train_with_wu=True, # Maximum 8 warm up epochs, then unfreeze the backbone (number 8 is just simply conclusion of our empirical experiments, feel free to adjust it)
                          acc_steps=4, # accumulation means a batch is technically 16* 4 = 64 
                          criterion=crit_aslcb, # ASL-CB eliminating negative class dominance and focus on positive samples, with class-balanced weights to further avoid head bias
                          name="trans_aslcb")
"""