import argparse
from configure import create_bitexts
from configure import create_bitexts_from_experiment_dir
from configure import USE_CUDA
import evaluate
import json
from myutil import load_model_for_inference, logger
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch


def translate(
    src_tokenized,
    tgt_tokenizer,
    model,
    tgt_lang,
    permutation=None,
    a=32,
    b=3,
    num_beams=4,
    **kwargs,
):
    """Beam-searches a translation for one batch of tokenized source sentences,
    un-enciphering the generated token ids first if permutation is given."""
    model.eval()
    src_tokenized = {k: v.to(model.device) for k, v in src_tokenized.items()}
    with torch.no_grad():
        result = model.generate(
            **src_tokenized,
            forced_bos_token_id=tgt_tokenizer.get_special_tokens()[tgt_lang],
            max_new_tokens=int(a + b * src_tokenized["input_ids"].shape[1]),
            num_beams=num_beams,
            **kwargs,
        )
    result = result.to("cpu")
    if permutation is not None:
        result.apply_(permutation.get_inverse())
    return tgt_tokenizer.batch_decode(result)


def translate_tokenized_mixture_of_bitexts(mix, model, tokenizer_map, cipher_map):
    """Translates every batch in a tokenized bitext mixture, grouping the
    resulting translations by source->target language pair."""
    if USE_CUDA:
        model.cuda()
    translations = dict()
    for batch in tqdm(mix, desc="Translating"):
        src, _, metadata = batch
        cipher = (
            cipher_map[metadata["lang2_tokenizer"], metadata["lang2_encipherment"]]
            if metadata["lang2_encipherment"] != "0"
            else None
        )
        src_code = metadata["lang1_code"]
        tgt_code = metadata["lang2_code"]
        key = "->".join([src_code, tgt_code])
        if key not in translations:
            translations[key] = []
        translated = translate(
            src,
            tokenizer_map[metadata["lang2_tokenizer"]],
            model,
            tgt_code,
            cipher,
        )
        translations[key].extend(translated)
        logger(f"translation: {translated[0]}")
    return translations


def evaluate_translations(candidate_translations, reference_translations):
    """Scores a list of candidate translations against their references using
    SacreBLEU and chrF."""
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result = bleu_calc.compute(
        predictions=candidate_translations, references=reference_translations
    )
    chrf_result = chrf_calc.compute(
        predictions=candidate_translations, references=reference_translations
    )
    return {
        "bleu": round(bleu_result["score"], 3),
        "chrf": round(chrf_result["score"], 3),
    }


def collate_references(test_data, tokenizer_map, cipher_map):
    """Decodes the target-side reference translations from a tokenized bitext
    mixture, un-enciphering them first if needed, grouped by source->target
    language pair."""
    references = dict()
    for _, tgt, metadata in tqdm(test_data, desc="References"):
        src_code = metadata["lang1_code"]
        tgt_code = metadata["lang2_code"]
        tgt_tokenizer = tokenizer_map[metadata["lang2_tokenizer"]]
        key = "->".join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        tgt_ids = tgt["input_ids"]
        tgt_ids[tgt_ids == -100] = 2  # TODO: make more general
        cipher = (
            cipher_map[metadata["lang2_tokenizer"], metadata["lang2_encipherment"]]
            if metadata["lang2_encipherment"] != "0"
            else None
        )
        if cipher is not None:
            tgt_ids.apply_(cipher.get_inverse())
        references[key].extend(tgt_tokenizer.batch_decode(tgt_ids))
    return references


def run_evaluation(bitexts, model_name_or_path, out_dir, suffix=""):
    """Translates and scores a model's test-set output, writing results into out_dir."""
    logger(f"Initializing model: {model_name_or_path}")
    use_fp16 = torch.cuda.is_available()
    model = load_model_for_inference(
        model_name_or_path, torch_dtype=torch.float16 if use_fp16 else torch.float32
    )
    if USE_CUDA:
        model.cuda()

    tokenizer_map = bitexts["tokenizer_map"]
    cipher_map = bitexts["cipher_map"]
    test_data = bitexts["test"]

    logger("Collating reference translations")
    test_data.restart()
    references = collate_references(test_data, tokenizer_map, cipher_map)
    with open(Path(out_dir) / "references.json", "w") as writer:
        json.dump(references, writer)
    logger("...references complete.")

    logger("Translating test data")
    test_data.restart()
    translations = translate_tokenized_mixture_of_bitexts(
        test_data, model, tokenizer_map, cipher_map
    )
    with open(Path(out_dir) / f"translations{suffix}.json", "w") as writer:
        json.dump(translations, writer)
    logger("...translation complete.")

    logger("Scoring translations")
    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open(Path(out_dir) / f"scores{suffix}.json", "w") as writer:
        json.dump(scores, writer, indent=2, ensure_ascii=False)
    logger("...scoring complete.")


def evaluate_experiment(experiment_dir, model_name=None, eval_batch_size=None):
    """Evaluates a finetuning experiment. By default, evaluates the experiment's
    own finetuned checkpoint; pass model_name to instead evaluate some other
    HuggingFace model (e.g. the un-finetuned base model) against the same test
    data, writing its outputs alongside the experiment's under a model-specific
    suffix so they don't clobber the experiment's own results."""
    bitexts = create_bitexts_from_experiment_dir(
        experiment_dir, eval_batch_size=eval_batch_size
    )
    if model_name is None:
        run_evaluation(bitexts, experiment_dir, experiment_dir)
    else:
        suffix = "." + model_name.replace("/", "_")
        run_evaluation(bitexts, model_name, experiment_dir, suffix=suffix)


def evaluate_config(config_file, model_name, out_dir, eval_batch_size=None):
    """Evaluates an arbitrary HuggingFace model against a config's test data,
    with no need for a pre-existing experiment directory."""
    with open(config_file) as reader:
        config = json.load(reader)
    if eval_batch_size is not None:
        config["finetuning_parameters"]["eval_batch_size"] = eval_batch_size
    bitexts = create_bitexts(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "." + model_name.replace("/", "_")
    run_evaluation(bitexts, model_name, out_dir, suffix=suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a translation model.")
    parser.add_argument("--dir", type=str, help="Experiment directory to evaluate.")
    parser.add_argument(
        "--config",
        type=str,
        help="Config file to evaluate against, for testing a model with no "
        "existing experiment directory. Requires --model and --out.",
    )
    parser.add_argument(
        "--out", type=str, help="Output directory (required with --config)."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name or path to evaluate. Defaults to the "
        "finetuned checkpoint in --dir. Required with --config.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Override the eval batch size from the config (decoupled from "
        "the training batch size).",
    )
    args = parser.parse_args()

    if args.dir is None and args.config is None:
        parser.error("Specify --dir or --config.")
    if args.config is not None:
        if args.model is None:
            parser.error("--model is required when using --config.")
        if args.out is None:
            parser.error("--out is required when using --config.")
        evaluate_config(
            args.config, args.model, args.out, eval_batch_size=args.eval_batch_size
        )
    else:
        evaluate_experiment(
            args.dir, model_name=args.model, eval_batch_size=args.eval_batch_size
        )
