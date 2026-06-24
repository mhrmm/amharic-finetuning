from configure import USE_CUDA
import gc
import sys
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from transformers import AutoTokenizer


def logger(s, to_stderr=False):
    if to_stderr:
        sys.stderr.write(str(s))
        sys.stderr.write("\n")
        sys.stderr.flush()
    else:
        sys.stdout.write(str(s))
        sys.stdout.write("\n")
        sys.stdout.flush()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def what_nllb_token_is_this(token_id, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    return tokenizer.convert_ids_to_tokens([token_id])[0]


def prepare_model_for_finetuning(ft_params):
    if ft_params.should_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(ft_params.base_model)
        print("loaded pretrained model")
    else:
        model_config = AutoConfig.from_pretrained(ft_params.base_model)
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        print("loaded architecture only")
    if hasattr(model.config, "max_length"):  # this should be in a GenerationConfig
        delattr(model.config, "max_length")
    if ft_params.freeze_decoder:
        print("--> DECODER FROZEN <--")
        for param in model.get_decoder().parameters():
            param.requires_grad = False
    else:
        print("--> decoder NOT frozen <--")
    if ft_params.freeze_encoder:
        print("--> ENCODER FROZEN <--")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    else:
        print("--> encoder NOT frozen <--")
    if ft_params.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=ft_params.lora_r,
            lora_alpha=ft_params.lora_alpha,
            lora_dropout=ft_params.lora_dropout,
            target_modules=ft_params.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        print(f"--> LoRA ENABLED (r={ft_params.lora_r}, alpha={ft_params.lora_alpha}) <--")
        model.print_trainable_parameters()
    if USE_CUDA:
        torch.cuda.set_device(0)
        model.cuda()
    return model


def merge_lora_checkpoint(experiment_dir, ft_params):
    """Merges a saved LoRA adapter checkpoint into its base model and
    overwrites experiment_dir with the merged full model, so it can be
    loaded downstream with plain AutoModelForSeq2SeqLM.from_pretrained."""
    base_model = AutoModelForSeq2SeqLM.from_pretrained(ft_params.base_model)
    model = PeftModel.from_pretrained(base_model, experiment_dir)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(experiment_dir)
