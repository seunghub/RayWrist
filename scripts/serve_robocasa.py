"""Serve a joint RayWrist checkpoint to the official GR1 Tabletop rollout client."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer
    from raywrist.integrations.robocasa import RoboCasaPolicy
    policy = RoboCasaPolicy.from_pretrained(args.base_model, args.checkpoint, device=args.device)
    policy = Gr00tSimPolicyWrapper(policy)
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == '__main__':
    main()
