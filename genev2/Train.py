import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, average_precision_score
import wandb
import itertools

def train_val_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs,
    checkpoint_prefix="model",
    use_wandb: bool = False,
    max_grad_norm: float = None,
    scheduler=None              # e.g. a warm‐up or LR scheduler
):
    """
    Trains and validates the model, tracking per‐batch metrics and:
      - gradient clipping
      - NaN checks
      - wandb logging of loss, metrics, weight & grad norms

    Args:
        use_wandb: if True, logs metrics to Weights & Biases
        max_grad_norm: clip gradients to this L2 norm (optional)
        scheduler: lr scheduler (should be stepped after optimizer.step)
    """
    history = {
        'train': {k: [] for k in [
            'loss', 'true_pos', 'true_neg', 'pred_pos', 'pred_neg',
            'prec_pos', 'rec_pos', 'prec_neg', 'rec_neg',
            'ap_pos', 'ap_neg'
        ]},
        'val':   {k: [] for k in [
            'loss', 'true_pos', 'true_neg', 'pred_pos', 'pred_neg',
            'prec_pos', 'rec_pos', 'prec_neg', 'rec_neg',
            'ap_pos', 'ap_neg'
        ]}
    }

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_iter = iter(train_loader)
        val_iter   = itertools.cycle(val_loader)
        num_batches = len(train_loader)

        for batch_idx in range(num_batches):
            # --- Training Batch ---
            batch = next(train_iter)
            inputs = {
                'type_ids':    batch['type_track'].to(device),
                'biotype_ids': batch['biotype_track'].to(device),
                'strand_ids':  batch['strand_track'].to(device),
                'homologs':    batch['homologs_track'].to(device),
                'protein_emb': batch['protein_embeddings'].to(device),
                'dna_emb':     batch['dna_embeddings'].to(device),
                'graph_emb':   batch['graph_embeddings'].to(device),
                'mask':        batch['padding_mask'].to(device)
            }
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(
                inputs['type_ids'], inputs['biotype_ids'], inputs['strand_ids'],
                inputs['homologs'], inputs['protein_emb'], inputs['dna_emb'], inputs['graph_emb'],
                mask=inputs['mask']
            )
            loss = loss_fn(logits, labels, inputs['mask'])
            if torch.isnan(loss):
                raise RuntimeError(f"NaN in loss at epoch {epoch}, batch {batch_idx}")

            loss.backward()

            # Gradient clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Check for NaNs in grads
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    raise RuntimeError(f"NaN in grad for {name} at epoch {epoch}, batch {batch_idx}")

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Compute metrics
            B, L, C = logits.shape
            probs = F.softmax(logits, dim=-1).view(-1, C)
            flat_labels = labels.view(-1)
            flat_mask   = inputs['mask'].view(-1)
            valid = (flat_labels != -1) & (~flat_mask)
            y_true   = flat_labels[valid].cpu().numpy()
            y_scores = probs[valid].detach().cpu().numpy()
            y_pred   = logits.argmax(dim=-1).view(-1)[valid].cpu().numpy()

            true_pos = (y_true == 1).sum()
            true_neg = (y_true == 0).sum()
            pred_pos = (y_pred == 1).sum()
            pred_neg = (y_pred == 0).sum()
            prec_pos = precision_score(y_true == 1, y_pred == 1, zero_division=0)
            rec_pos  = recall_score   (y_true == 1, y_pred == 1, zero_division=0)
            prec_neg = precision_score(y_true == 0, y_pred == 0, zero_division=0)
            rec_neg  = recall_score   (y_true == 0, y_pred == 0, zero_division=0)
            ap_pos   = average_precision_score(y_true == 1, y_scores[:,1])
            ap_neg   = average_precision_score(y_true == 0, y_scores[:,0])

            # Save to history
            h_train = history['train']
            for k, v in zip(
                ['loss','true_pos','true_neg','pred_pos','pred_neg',
                 'prec_pos','rec_pos','prec_neg','rec_neg','ap_pos','ap_neg'],
                [loss.item(),true_pos,true_neg,pred_pos,pred_neg,
                 prec_pos,rec_pos,prec_neg,rec_neg,ap_pos,ap_neg]
            ):
                h_train[k].append(v)

            # wandb logging
            if use_wandb:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/prec_pos': prec_pos, 'train/rec_pos': rec_pos,
                    'train/prec_neg': prec_neg, 'train/rec_neg': rec_neg
                }
                # weight & grad norms
                for name, param in model.named_parameters():
                    log_dict[f'weights/{name}_norm'] = param.norm().item()
                    if param.grad is not None:
                        log_dict[f'grads/{name}_norm'] = param.grad.norm().item()
                wandb.log(log_dict, step=global_step)

            print(
                f"[Epoch {epoch:02d} Batch {batch_idx+1}/{num_batches}] "
                f"TRAIN loss={loss.item():.4f} "
                f"T+={true_pos} T-={true_neg} P+={pred_pos} P-={pred_neg} "
                f"Prec+={prec_pos:.3f} Rec+={rec_pos:.3f} "
                f"Prec-={prec_neg:.3f} Rec-={rec_neg:.3f} "
                f"AP+={ap_pos:.3f} AP-={ap_neg:.3f}"
            )

            global_step += 1

            # --- Validation Batch ---
            model.eval()
            with torch.no_grad():
                batch = next(val_iter)
                inputs = {
                    'type_ids':    batch['type_track'].to(device),
                    'biotype_ids': batch['biotype_track'].to(device),
                    'strand_ids':  batch['strand_track'].to(device),
                    'homologs':    batch['homologs_track'].to(device),
                    'protein_emb': batch['protein_embeddings'].to(device),
                    'dna_emb':     batch['dna_embeddings'].to(device),
                    'graph_emb':   batch['graph_embeddings'].to(device),
                    'mask':        batch['padding_mask'].to(device)
                }
                labels = batch['labels'].to(device)

                logits = model(
                    inputs['type_ids'], inputs['biotype_ids'], inputs['strand_ids'],
                    inputs['homologs'], inputs['protein_emb'], inputs['dna_emb'], inputs['graph_emb'],
                    mask=inputs['mask']
                )
                val_loss = loss_fn(logits, labels, inputs['mask'])
                if torch.isnan(val_loss):
                    raise RuntimeError(f"NaN in val loss at epoch {epoch}, batch {batch_idx}")

                B, L, C = logits.shape
                probs = F.softmax(logits, dim=-1).view(-1, C)
                flat_labels = labels.view(-1)
                flat_mask   = inputs['mask'].view(-1)
                valid = (flat_labels != -1) & (~flat_mask)
                y_true   = flat_labels[valid].cpu().numpy()
                y_scores = probs[valid].detach().cpu().numpy()
                y_pred   = logits.argmax(dim=-1).view(-1)[valid].cpu().numpy()

                true_pos = (y_true == 1).sum()
                true_neg = (y_true == 0).sum()
                pred_pos = (y_pred == 1).sum()
                pred_neg = (y_pred == 0).sum()
                prec_pos = precision_score(y_true == 1, y_pred == 1, zero_division=0)
                rec_pos  = recall_score   (y_true == 1, y_pred == 1, zero_division=0)
                prec_neg = precision_score(y_true == 0, y_pred == 0, zero_division=0)
                rec_neg  = recall_score   (y_true == 0, y_pred == 0, zero_division=0)
                ap_pos   = average_precision_score(y_true == 1, y_scores[:,1])
                ap_neg   = average_precision_score(y_true == 0, y_scores[:,0])

                h_val = history['val']
                for k, v in zip(
                    ['loss','true_pos','true_neg','pred_pos','pred_neg',
                     'prec_pos','rec_pos','prec_neg','rec_neg','ap_pos','ap_neg'],
                    [val_loss.item(),true_pos,true_neg,pred_pos,pred_neg,
                     prec_pos,rec_pos,prec_neg,rec_neg,ap_pos,ap_neg]
                ):
                    h_val[k].append(v)

                if use_wandb:
                    wandb.log({
                        'val/loss': val_loss.item(),
                        'val/prec_pos': prec_pos, 'val/rec_pos': rec_pos,
                        'val/prec_neg': prec_neg, 'val/rec_neg': rec_neg,
                        'val/lr': optimizer.param_groups[0]['lr']
                    }, step=global_step)

                print(
                    f"[Epoch {epoch:02d} Batch {batch_idx+1}/{num_batches}] "
                    f" VAL loss={val_loss.item():.4f} "
                    f"T+={true_pos} T-={true_neg} P+={pred_pos} P-={pred_neg} "
                    f"Prec+={prec_pos:.3f} Rec+={rec_pos:.3f} "
                    f"Prec-={prec_neg:.3f} Rec-={rec_neg:.3f} "
                    f"AP+={ap_pos:.3f} AP-={ap_neg:.3f}"
                )

            model.train()

        # end-of-epoch checkpoint  
        ckpt_path = f"{checkpoint_prefix}{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
        }, ckpt_path)
        print(f">>> Epoch {epoch} complete. Saved checkpoint: {ckpt_path}")

    return history
