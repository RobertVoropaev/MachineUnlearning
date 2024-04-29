import torch


def validate_model(model, dataloader, score_f, device):
    model.eval()
    
    mean_score = 0
    batches_n = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)

            mean_score += score_f(pred, targets)
            batches_n += 1
            
    mean_score /= batches_n
    return float(mean_score)
