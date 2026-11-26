"""Run one native action chunk from a recorded GR1 head observation."""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--observation', required=True, help='NPZ containing head_rgb and physical state44')
    parser.add_argument('--prompt', required=True, help='Native task instruction, including its waist prefix')
    parser.add_argument('--output', default='actions.npz')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from raywrist.integrations.robocasa import RoboCasaPolicy
    policy = RoboCasaPolicy.from_pretrained(args.base_model, args.checkpoint, device=args.device)
    with np.load(args.observation, allow_pickle=False) as observation:
        actions = policy.infer(observation['state44'], observation['head_rgb'], args.prompt)
    np.savez(args.output, **actions)
    for key, value in actions.items():
        print(key, value.shape)


if __name__ == '__main__':
    main()
