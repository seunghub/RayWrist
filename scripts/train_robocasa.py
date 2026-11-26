"""Joint head-camera-only training with the native GR00T dataset and Trainer."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True, help='Native GR00T N1.7 base checkpoint')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/robocasa_gr1.json')
    parser.add_argument('--baseline', action='store_true', help='Train native head-only FT without readers')
    parser.add_argument('--dry-run', action='store_true', help='Show resolved task paths and budget without loading a model')
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    recipe = spec['training']
    paths = [args.dataset_root / name for name in spec['dataset_directories']]
    plan = {
        'condition': 'native_ft' if args.baseline else 'ft_raywrist',
        'dataset_paths': [str(path) for path in paths],
        'updates': recipe['updates'], 'global_batch': recipe['global_batch'],
        'microbatch_per_device': recipe['microbatch_per_device'],
        'accumulation': recipe['accumulation'], 'devices': recipe['devices'],
        'wrist_teacher': False, 'alignment_weight': 0,
    }
    spec['condition'] = plan['condition']
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if len(paths) != 24 or any(not (path / 'meta/info.json').is_file() for path in paths):
        parser.error('dataset-root must contain all 24 prepared GR1 Tabletop task directories')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != recipe['devices'] or world * recipe['microbatch_per_device'] * recipe['accumulation'] != recipe['global_batch']:
        parser.error('Launch four ranks for the 32 x 4 x 4 = 512 global batch recipe')

    import torch
    import torch.distributed as dist
    from transformers import TrainingArguments, TrainerCallback, set_seed
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.data.data_config import SingleDatasetConfig
    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.experiment.trainer import Gr00tTrainer
    from raywrist.integrations.robocasa import (
        TAG, dataset_modalities, load_model, processor_from_checkpoint,
    )

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    if not dist.is_initialized():
        dist.init_process_group('nccl')
    set_seed(recipe['training_seed'])
    processor = processor_from_checkpoint(args.base_model, enabled=not args.baseline)
    model = load_model(args.base_model, enabled=not args.baseline)
    config = get_default_config()
    config.data.datasets = [SingleDatasetConfig(dataset_paths=[str(p) for p in paths], embodiment_tag=TAG)]
    config.data.modality_configs = dataset_modalities(processor)
    config.data.override_pretraining_statistics = False
    for key, value in spec['data'].items():
        setattr(config.data, key, deepcopy(value))
    config.training.eval_strategy = 'no'
    dataset, _ = DatasetFactory(config).build(processor=processor)
    training = TrainingArguments(
        output_dir=str(args.output_dir), max_steps=recipe['updates'],
        per_device_train_batch_size=recipe['microbatch_per_device'],
        gradient_accumulation_steps=recipe['accumulation'],
        learning_rate=recipe['peak_lr'], lr_scheduler_type='cosine',
        warmup_steps=recipe['warmup_updates'], weight_decay=recipe['weight_decay'],
        adam_beta1=recipe['betas'][0], adam_beta2=recipe['betas'][1], adam_epsilon=recipe['eps'],
        max_grad_norm=recipe['gradient_clip'], optim='adamw_torch',
        bf16=True, tf32=True, seed=recipe['training_seed'],
        dataloader_num_workers=4, remove_unused_columns=False,
        save_strategy='steps', save_steps=1000, save_total_limit=3,
        eval_strategy='no', logging_steps=10, report_to=[],
        ddp_find_unused_parameters=True, ddp_broadcast_buffers=False,
        gradient_checkpointing=False,
    )

    class SaveProcessor(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                folder = Path(args.output_dir) / f'checkpoint-{state.global_step}'
                processor.save_pretrained(folder)
                (folder / 'raywrist_robocasa.json').write_text(json.dumps(spec, indent=2) + '\n')

    trainer = Gr00tTrainer(model=model, args=training, train_dataset=dataset,
                          data_collator=processor.collator, callbacks=[SaveProcessor()])
    trainer.train()
    if trainer.state.global_step != recipe['updates']:
        raise RuntimeError('Training did not reach the scheduled final update')
    final = args.output_dir / 'final30000'
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        processor.save_pretrained(final)
        saved = deepcopy(spec)
        saved.update(condition=plan['condition'], actual_optimizer_updates=trainer.state.global_step)
        (final / 'raywrist_robocasa.json').write_text(json.dumps(saved, indent=2) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
