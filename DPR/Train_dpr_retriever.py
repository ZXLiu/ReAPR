#  built-in
import math, logging, json, random, functools, os
import types

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ["WANDB_IGNORE_GLOBS"] = '*.bin'  # not upload ckpt to wandb cloud

# third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
import transformers
from transformers import (
    BertTokenizer, RobertaTokenizer,
    BertModel, RobertaForMaskedLM,
    AutoModel, AutoTokenizer
)

transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

# own
from utils import (
    get_yaml_file,
    set_seed,
    get_linear_scheduler,
)

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # adding args here for more control from CLI is possible
    parser.add_argument("--config_file", default='train_dpr_retriever.yaml')
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args


class DualEncoder(nn.Module):
    def __init__(self, bug_encoder, fix_encoder):
        super().__init__()
        self.bug_encoder = bug_encoder
        self.fix_encoder = fix_encoder

    def forward(
            self,
            bug_input_ids,  # [bs,seq_len]
            bug_attention_mask,  # [bs,seq_len]
            fix_input_ids,  # [bs*n_doc,seq_len]
            fix_attention_mask,  # [bs*n_doc,seq_len]
    ):
        CLS_POS = 0
        # [bs,n_dim]
        bug_embedding = self.bug_encoder(
            input_ids=bug_input_ids,
            attention_mask=bug_attention_mask,
        ).last_hidden_state[:,CLS_POS,:]

        # [bs * n_doc,n_dim]
        fix_embedding = self.fix_encoder(
            input_ids=fix_input_ids,
            attention_mask=fix_attention_mask,
        ).last_hidden_state[:,CLS_POS,:]

        return bug_embedding, fix_embedding


def calculate_dpr_loss(matching_score, labels):
    return F.nll_loss(input=F.log_softmax(matching_score, dim=1), target=labels)


def calculate_hit_cnt(matching_score, labels):
    _, max_ids = torch.max(matching_score, 1)
    return (max_ids == labels).sum()


def calculate_average_rank(matching_score, labels):
    _, indices = torch.sort(matching_score, dim=1, descending=True)
    ranks = []
    for idx, label in enumerate(labels):
        rank = ((indices[idx] == label).nonzero()).item() + 1  # rank starts from 1
        ranks.append(rank)
    return ranks


class APRDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.data = json.load(open(file_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples, tokenizer):

        # prepare bug input
        bugs = [x['buggy_function'] for x in samples]
        bugs_inputs = tokenizer(bugs, max_length=256, padding=True, truncation=True, return_tensors='pt')

        # prepare fix input
        fixes = [x['fixed_function'] for x in samples]
        fixes_inputs = tokenizer(fixes, max_length=256, padding=True, truncation=True, return_tensors='pt')

        return {
            'bug_input_ids': bugs_inputs.input_ids,
            'bug_attention_mask': bugs_inputs.attention_mask,

            "fix_input_ids": fixes_inputs.input_ids,
            "fix_attention_mask": fixes_inputs.attention_mask,
        }


def validate(model, dataloader, accelerator):
    model.eval()
    bug_embeddings = []
    fix_embeddings = []
    for batch in dataloader:
        with torch.no_grad():
            bug_embedding, fix_embedding = model(**batch)
        bug_embeddings.append(bug_embedding.cpu())
        fix_embeddings.append(fix_embedding.cpu())

    bug_embeddings = torch.cat(bug_embeddings, dim=0)
    fix_embeddings = torch.cat(fix_embeddings, dim=0)
    matching_score = torch.matmul(bug_embeddings, fix_embeddings.permute(1, 0))  # bs, num_pos+num_neg
    labels = torch.arange(bug_embeddings.shape[0], dtype=torch.int64).to(matching_score.device)
    loss = calculate_dpr_loss(matching_score, labels=labels).item()
    ranks = calculate_average_rank(matching_score, labels=labels)

    if accelerator.use_distributed and accelerator.num_processes > 1:
        ranks_from_all_gpus = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(ranks_from_all_gpus, ranks)
        ranks = [x for y in ranks_from_all_gpus for x in y]

        loss_from_all_gpus = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(loss_from_all_gpus, loss)
        loss = sum(loss_from_all_gpus) / len(loss_from_all_gpus)

    return sum(ranks) / len(ranks), loss


def main():
    args = parse_args()
    set_seed(args.seed)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        mixed_precision='no',
        kwargs_handlers=[kwargs]
    )

    accelerator.init_trackers(
        project_name="dpr_for_APR_GraphCodeBERT",
        config=args,
    )
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir

    # path = os.path.dirname(os.path.abspath(__file__)) + "/GraphCodeBERT"
    path = "/home/lzx/LLM Model/CodeBERT"
    tokenizer = AutoTokenizer.from_pretrained(path)
    bug_encoder = AutoModel.from_pretrained(path)
    fix_encoder = AutoModel.from_pretrained(path)
    dual_encoder = DualEncoder(bug_encoder, fix_encoder)
    dual_encoder.train()

    train_dataset = APRDataset(args.train_file)
    train_collate_fn = functools.partial(APRDataset.collate_fn, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                                                   shuffle=True, collate_fn=train_collate_fn, num_workers=4,
                                                   pin_memory=True)

    dev_dataset = APRDataset(args.dev_file)
    dev_collate_fn = functools.partial(APRDataset.collate_fn, tokenizer=tokenizer)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False,
                                                 collate_fn=dev_collate_fn, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(dual_encoder.parameters(), lr=args.lr, eps=args.adam_eps)

    dual_encoder, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        dual_encoder, optimizer, train_dataloader, dev_dataloader,
    )

    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval, int) else int(
        args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = get_linear_scheduler(optimizer, warmup_steps=args.warmup_steps, total_training_steps=MAX_TRAIN_STEPS)

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process, ncols=100)

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed + epoch)
        progress_bar.set_description(f"epoch: {epoch + 1}/{MAX_TRAIN_EPOCHS}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dual_encoder):
                with accelerator.autocast():
                    bug_embedding, fix_embedding = dual_encoder(**batch)
                    single_device_bug_num, _ = bug_embedding.shape
                    single_device_fix_num, _ = fix_embedding.shape
                    if accelerator.use_distributed:
                        fix_list = [torch.zeros_like(fix_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=fix_list, tensor=fix_embedding.contiguous())
                        fix_list[dist.get_rank()] = fix_embedding
                        fix_embedding = torch.cat(fix_list, dim=0)

                        bug_list = [torch.zeros_like(bug_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=bug_list, tensor=bug_embedding.contiguous())
                        bug_list[dist.get_rank()] = bug_embedding
                        bug_embedding = torch.cat(bug_list, dim=0)

                    matching_score = torch.matmul(bug_embedding, fix_embedding.permute(1, 0))
                    labels = torch.cat(
                        [torch.arange(single_device_bug_num) + gpu_index * single_device_fix_num for gpu_index in
                         range(accelerator.num_processes)], dim=0).to(matching_score.device)
                    loss = calculate_dpr_loss(matching_score, labels=labels)

                accelerator.backward(loss)

                # one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
                    completed_steps += 1
                    accelerator.clip_grad_norm_(dual_encoder.parameters(), args.max_grad_norm)
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)

                    if completed_steps % EVAL_STEPS == 0:
                        avg_rank, loss = validate(dual_encoder, dev_dataloader, accelerator)
                        dual_encoder.train()
                        accelerator.log({"avg_rank": avg_rank, "loss": loss}, step=completed_steps)
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(dual_encoder)
                            unwrapped_model.bug_encoder.save_pretrained(
                                os.path.join(LOG_DIR, f"step-{completed_steps}/bug_encoder"))
                            tokenizer.save_pretrained(os.path.join(LOG_DIR, f"step-{completed_steps}/bug_encoder"))

                            unwrapped_model.fix_encoder.save_pretrained(
                                os.path.join(LOG_DIR, f"step-{completed_steps}/fix_encoder"))
                            tokenizer.save_pretrained(os.path.join(LOG_DIR, f"step-{completed_steps}/fix_encoder"))

                        accelerator.wait_for_everyone()

                optimizer.step()
                optimizer.zero_grad()

    if accelerator.is_local_main_process: wandb_tracker.finish()
    accelerator.end_training()


if __name__ == '__main__':
    main()
