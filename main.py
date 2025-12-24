import argparse
import logging
import os
from pathlib import Path
import random
import time
from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from nltk.translate.bleu_score import sentence_bleu

from module.model import BiLSTM, Transformer, CNN, BiLSTMAttn, BiLSTMCNN, BiLSTMConvAttRes
from module import Tokenizer, init_model_by_key
from module.metric import calc_bleu, calc_rouge_l
from module.decoding import DecodeOptions, decode_with_options

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(m: nn.Module) -> nn.Module:
    return cast(nn.Module, m.module) if hasattr(m, 'module') else m


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=768, type=int)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("-m", "--model", default='transformer', type=str)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16_opt_level", default='O1', type=str)
    parser.add_argument("--max_grad_norm", default=3.0, type=float)
    parser.add_argument("--dir", default='dataset', type=str)
    parser.add_argument("--output", default='output', type=str)
    parser.add_argument("--logdir", default='runs', type=str)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--n_layer", default=1, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--ff_dim", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)

    parser.add_argument("--test_epoch", default=1, type=int)
    parser.add_argument("--save_epoch", default=10, type=int)

    parser.add_argument("--embed_drop", default=0.2, type=float)
    parser.add_argument("--hidden_drop", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--ddp", action='store_true', help='Use DistributedDataParallel (launch with torchrun)')
    parser.add_argument("--dp", action='store_true', help='Use DataParallel (legacy, not recommended)')
    parser.add_argument("--resume", default=None, type=str, help='Resume training from checkpoint path')
    parser.add_argument("--early_stop_patience", default=0, type=int, help='Early stopping patience (0 disables)')
    parser.add_argument("--early_stop_min_delta", default=0.0, type=float, help='Minimum improvement for early stopping')

    parser.add_argument("--decode", default='constrained', choices=['argmax', 'constrained', 'beam'], type=str)
    parser.add_argument("--decode_topk", default=20, type=int)
    parser.add_argument("--decode_beam_size", default=5, type=int)
    parser.add_argument("--decode_no_copy", action='store_true', default=False)
    parser.add_argument("--decode_max_repeat", default=2, type=int)
    parser.add_argument("--decode_match_punct", action='store_true', default=False)
    return parser.parse_args()


def evaluate_loss(model, dataloader, loss_function, tokenizer):
    device = next(model.parameters()).device
    model.eval()
    sum_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, lens, target_ids = batch
            logits = model(input_ids, masks)
            loss = loss_function(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
            if loss.numel() > 1:
                loss = loss.mean()
            sum_loss += float(loss.item())
            steps += 1
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([sum_loss, float(steps)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_loss, steps = t.tolist()
        steps = int(steps)
    return sum_loss / max(steps, 1)

def auto_evaluate(model, testloader, tokenizer):
    sum_bleu = 0.0
    sum_rl = 0.0
    count = 0
    device = next(model.parameters()).device
    model.eval()
    for step, batch in enumerate(testloader):
        input_ids, masks, lens = tuple(t.to(device) for t in batch[:-1])
        target_ids = batch[-1]
        with torch.no_grad():
            logits = model(input_ids, masks)
            # preds.shape=(batch_size, max_seq_len)
            _, preds = torch.max(logits, dim=-1) 
        for seq, tag in zip(preds.tolist(), target_ids.tolist()):
            seq = list(filter(lambda x: x != tokenizer.pad_id, seq))
            tag = list(filter(lambda x: x != tokenizer.pad_id, tag))
            bleu = float(calc_bleu(seq, tag))
            rl = float(calc_rouge_l(seq, tag))
            sum_bleu += bleu
            sum_rl += rl
            count += 1
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([sum_bleu, sum_rl, float(count)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_bleu, sum_rl, count = t.tolist()
        count = int(count)
    return sum_bleu / max(count, 1), sum_rl / max(count, 1)

def predict_demos(model, tokenizer:Tokenizer, decode_options: DecodeOptions):
    demos = [
        "马齿草焉无马齿", "天古天今，地中地外，古今中外存天地", 
        "笑取琴书温旧梦", "日里千人拱手划船，齐歌狂吼川江号子",
        "我有诗情堪纵酒", "我以真诚溶冷血",
        "三世业岐黄，妙手回春人共赞"
    ]
    input_ids_list = [tokenizer.encode(sent) for sent in demos]
    sents = [torch.tensor(ids).unsqueeze(0) for ids in input_ids_list]
    model.eval()
    device = next(model.parameters()).device
    for i, sent in enumerate(sents):
        sent = sent.to(device)
        with torch.no_grad():
            logits = model(sent).squeeze(0)
        pred_ids = decode_with_options(logits, input_ids_list[i], tokenizer, decode_options)
        pred = tokenizer.decode(pred_ids)
        logger.info(f"上联：{demos[i]}。 预测的下联：{pred}")

def save_model(filename, model: nn.Module, args, tokenizer, optimizer=None, scheduler=None, scaler=None, epoch=None, global_step=None):
    model_to_save = unwrap_model(model)
    info_dict = {
        'model': model_to_save.state_dict(),
        'args': args,
        'tokenizer': tokenizer
    }
    if optimizer is not None:
        info_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        info_dict['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        info_dict['scaler'] = scaler.state_dict()
    if epoch is not None:
        info_dict['epoch'] = epoch
    if global_step is not None:
        info_dict['global_step'] = global_step
    torch.save(info_dict, filename)

def run():
    args = get_args()
    set_seed(args.seed)
    fdir = Path(args.dir)
    local_rank = 0
    rank = 0
    world_size = 1
    if args.ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        backend = 'nccl' if torch.cuda.is_available() and not args.no_cuda and dist.is_nccl_available() else 'gloo'
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available() and not args.no_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    is_main_process = (not args.ddp) or rank == 0
    tb = SummaryWriter(args.logdir) if is_main_process else None
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    if is_main_process:
        logger.info(args)
    logger.info(f"loading vocab...")
    tokenizer = Tokenizer.from_pretrained(fdir / 'vocab.pkl')
    logger.info(f"loading dataset...")
    train_dataset = torch.load(fdir / 'train.pkl', weights_only=False)
    test_dataset = torch.load(fdir / 'test.pkl', weights_only=False)
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    logger.info(f"initializing model...")
    model: nn.Module = init_model_by_key(args, tokenizer)
    model.to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=bool(args.fp16 and device.type == 'cuda'))
    if args.ddp:
        model = cast(nn.Module, DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None))
    elif args.dp and torch.cuda.device_count() > 1:
        model = cast(nn.Module, torch.nn.DataParallel(model))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    if is_main_process:
        logger.info(f"num gpu: {torch.cuda.device_count()} (ddp={args.ddp}, world_size={world_size})")

    decode_opts = DecodeOptions(
        strategy=args.decode,
        topk=args.decode_topk,
        beam_size=args.decode_beam_size,
        no_copy=bool(args.decode_no_copy),
        max_repeat=int(args.decode_max_repeat),
        match_punct=bool(args.decode_match_punct),
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    no_improve_epochs = 0

    if args.resume is not None:
        if is_main_process:
            logger.info(f"resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model_to_load = unwrap_model(model)
        model_to_load.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and scaler is not None:
            try:
                scaler.load_state_dict(ckpt['scaler'])
            except Exception:
                pass
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        global_step = int(ckpt.get('global_step', 0))
        best_val_loss = float(ckpt.get('best_val_loss', best_val_loss))
        no_improve_epochs = int(ckpt.get('no_improve_epochs', no_improve_epochs))

    for epoch in range(start_epoch, args.epochs):
        if args.ddp:
            train_sampler.set_epoch(epoch)
        if is_main_process:
            logger.info(f"***** Epoch {epoch} *****")
        model.train()
        t1 = time.time()
        sum_train_loss = 0.0
        train_steps = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, lens, target_ids = batch
            with autocast(enabled=scaler.is_enabled()):
                logits = model(input_ids, masks)
                loss = loss_function(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            sum_train_loss += float(loss.item())
            train_steps += 1
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            if step % 100 == 0:
                if tb is not None:
                    tb.add_scalar('loss', loss.item(), global_step)
                if is_main_process:
                    logger.info(
                        f"[epoch]: {epoch}, [batch]: {step}, [loss]: {loss.item()}")
            global_step += 1

        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([sum_train_loss, float(train_steps)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            sum_train_loss, train_steps = t.tolist()
            train_steps = int(train_steps)

        mean_train_loss = sum_train_loss / max(train_steps, 1)
        scheduler.step(mean_train_loss)
        t2 = time.time()
        if is_main_process:
            logger.info(f"epoch time: {t2-t1:.5}, mean train loss: {mean_train_loss:.6}")

        val_loss = None
        if (epoch + 1) % args.test_epoch == 0:
            if is_main_process:
                predict_demos(model, tokenizer, decode_opts)
            val_loss = evaluate_loss(model, test_loader, loss_function, tokenizer)
            bleu, rl = auto_evaluate(model, test_loader, tokenizer)
            if is_main_process:
                logger.info(f"val loss: {val_loss:.6}")
                logger.info(f"BLEU: {round(bleu, 9)}, Rouge-L: {round(rl, 8)}")

        if args.early_stop_patience > 0 and val_loss is not None:
            improved = (best_val_loss - val_loss) > args.early_stop_min_delta
            if improved:
                best_val_loss = val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if is_main_process:
                logger.info(f"early stop: best_val_loss={best_val_loss:.6}, no_improve_epochs={no_improve_epochs}")
            if no_improve_epochs >= args.early_stop_patience:
                if is_main_process:
                    logger.info("early stopping triggered")
                break

        if (epoch + 1) % args.save_epoch == 0:
            if is_main_process:
                filename = f"{model.__class__.__name__}_{epoch + 1}.bin"
                filename = output_dir / filename
                save_model(
                    filename,
                    model,
                    args,
                    tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                )

                # Save an extra small training-state checkpoint for resume convenience
                state_path = output_dir / "last_state.bin"
                extra = {
                    'model': unwrap_model(model).state_dict(),
                    'args': args,
                    'tokenizer': tokenizer,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'epoch': epoch,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss,
                    'no_improve_epochs': no_improve_epochs,
                }
                torch.save(extra, state_path)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    run()