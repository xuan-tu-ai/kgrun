"""
Command-line interface entry point for kgrun.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from kgrun.config import CONFIG_FILE_NAME, create_default_config, find_config
from kgrun.data import push_dataset
from kgrun.runner import (
    get_authenticated_username,
    pull_outputs,
    run_script,
    show_logs,
    show_status,
)


def handle_init(args: argparse.Namespace, extra_args: list[str]) -> int:
    """
    Initialize kgrun.toml in the current working directory.
    """
    cwd = Path.cwd()
    target_config = cwd / CONFIG_FILE_NAME
    if target_config.exists() and not args.force:
        print(f"[kgrun] Configuration already exists at {target_config}. Use --force to overwrite.")
        return 0

    username = get_authenticated_username() or "username"
    folder_slug = cwd.name.lower().replace(" ", "-").replace("_", "-")
    if len(folder_slug) < 5:
        folder_slug = f"{folder_slug}-job"
    kernel_slug = f"{username}/{folder_slug}"

    create_default_config(target_config, kernel_slug=kernel_slug)
    print(f"[kgrun] Created {target_config} with kernel '{kernel_slug}'")

    # Verify Kaggle CLI connectivity
    try:
        check = subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "--page-size", "1"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode == 0:
            print("[kgrun] Kaggle CLI is installed and authentication is verified.")
        else:
            print(f"[kgrun] Warning: Kaggle CLI returned non-zero code. Output:\n{check.stderr.strip()}")
    except FileNotFoundError:
        print("[kgrun] Warning: 'kaggle' CLI not found on PATH. Please ensure it is installed.")

    return 0


def handle_run(args: argparse.Namespace, extra_args: list[str]) -> int:
    """
    Dispatch run command, forwarding any unparsed arguments to the target script.
    """
    # Clean leading '--' if present
    forwarded_args = list(extra_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    return run_script(
        script=args.script,
        script_args=forwarded_args,
        gpu=args.gpu,
        no_follow=args.no_follow,
        pull=args.pull,
        timeout_h=args.timeout,
    )


def handle_pull(args: argparse.Namespace, extra_args: list[str]) -> int:
    """
    Dispatch pull command.
    """
    return pull_outputs(pattern=args.pattern)


def handle_logs(args: argparse.Namespace, extra_args: list[str]) -> int:
    """
    Dispatch logs command.
    """
    return show_logs(follow=args.follow)


def handle_status(args: argparse.Namespace, extra_args: list[str]) -> int:
    """
    Dispatch status command.
    """
    return show_status()


def handle_data(args: argparse.Namespace, extra_args: list[str]) -> int:
    """
    Dispatch data commands.
    """
    if args.data_subcommand == "push":
        return push_dataset(
            dir_path=args.dir,
            message=args.message,
            slug=args.slug,
            update_config=not args.no_config,
        )
    print("[kgrun] Error: Missing data subcommand. See 'kgrun data --help'", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """
    Construct argument parser for CLI commands.
    """
    parser = argparse.ArgumentParser(
        prog="kgrun",
        description="Run local Python scripts seamlessly on remote Kaggle GPUs.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # init
    p_init = subparsers.add_parser("init", help="Initialize kgrun.toml in current directory")
    p_init.add_argument("--force", "-f", action="store_true", help="Overwrite existing kgrun.toml")
    p_init.set_defaults(func=handle_init)

    # run
    p_run = subparsers.add_parser("run", help="Run a script on Kaggle GPU")
    p_run.add_argument("script", help="Target Python script to execute")
    p_run.add_argument("--gpu", default=None, help="GPU accelerator (e.g. t4, p100, none)")
    p_run.add_argument("--no-follow", action="store_true", help="Do not follow live logs after push")
    p_run.add_argument("--pull", action="store_true", help="Automatically pull outputs after run completes")
    p_run.add_argument("--timeout", type=float, default=None, help="Timeout in hours")
    p_run.set_defaults(func=handle_run)

    # pull
    p_pull = subparsers.add_parser("pull", help="Download latest kernel outputs to runs/<timestamp>/")
    p_pull.add_argument("--pattern", default=None, help="File regex pattern filter")
    p_pull.set_defaults(func=handle_pull)

    # logs
    p_logs = subparsers.add_parser("logs", help="Display execution logs for current kernel")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow live execution logs")
    p_logs.set_defaults(func=handle_logs)

    # status
    p_status = subparsers.add_parser("status", help="Check status of latest kernel execution")
    p_status.set_defaults(func=handle_status)

    # data
    p_data = subparsers.add_parser("data", help="Manage Kaggle datasets")
    data_sub = p_data.add_subparsers(dest="data_subcommand", help="Data operations")
    p_data_push = data_sub.add_parser("push", help="Create or version a Kaggle dataset from local directory")
    p_data_push.add_argument("dir", help="Directory path to upload as dataset")
    p_data_push.add_argument("-m", "--message", default=None, help="Dataset version notes")
    p_data_push.add_argument("--slug", default=None, help="Dataset slug (e.g. username/dataset-name)")
    p_data_push.add_argument("--no-config", action="store_true", help="Do not update kgrun.toml with dataset slug")
    p_data.set_defaults(func=handle_data)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI main entry point.
    """
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if not args.subcommand:
        parser.print_help()
        return 0
    return args.func(args, unknown)


if __name__ == "__main__":
    sys.exit(main())
