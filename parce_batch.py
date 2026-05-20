#!/usr/bin/env python3
"""
RCNet Batch Training Script

This script works with create_rcnet_folders.py to support batch execution of generated training environments.
"""

import os
import sys
import subprocess
import concurrent.futures
import argparse
import time
from datetime import datetime
import logging
from pathlib import Path
import threading
import queue


def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_train_{timestamp}.log"

    # Prevent duplicate handlers
    logger = logging.getLogger("batch_train")
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def get_available_gpus():
    """Get list of available GPUs"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.getLogger(__name__).warning("Cannot detect GPU, running in CPU mode")
        return []


class GPUManager:
    """GPU resource manager"""

    def __init__(self, available_gpus=None):
        self.logger = logging.getLogger(__name__)
        self.available_gpus = available_gpus if available_gpus is not None else get_available_gpus()
        self.gpu_queue = queue.Queue()
        for gpu_id in self.available_gpus:
            self.gpu_queue.put(gpu_id)
        self.logger.info(f"Detected {len(self.available_gpus)} GPUs: {self.available_gpus}")

    def acquire_gpu(self, timeout=None):
        try:
            gpu_id = self.gpu_queue.get(timeout=timeout)
            self.logger.info(f"Allocated GPU {gpu_id}")
            return gpu_id
        except queue.Empty:
            self.logger.warning("No available GPU")
            return None

    def release_gpu(self, gpu_id):
        if gpu_id is not None:
            self.gpu_queue.put(gpu_id)
            self.logger.info(f"Released GPU {gpu_id}")


def find_train_scripts(rcnet_training_dir, main_folders=None, num_folders=None):
    """Find all train.py script paths"""
    train_scripts = []
    if main_folders is None:
        main_folders = sorted([
            f for f in os.listdir(rcnet_training_dir)
            if os.path.isdir(os.path.join(rcnet_training_dir, f))
        ])

    for main_folder in main_folders:
        main_path = os.path.join(rcnet_training_dir, main_folder)
        if not os.path.exists(main_path):
            continue

        available_num_folders = sorted([
            f for f in os.listdir(main_path)
            if os.path.isdir(os.path.join(main_path, f)) and f.isdigit()
        ], key=int)

        if num_folders is not None:
            num_folders_str = [str(f) for f in num_folders]
            available_num_folders = [f for f in available_num_folders if f in num_folders_str]

        for num_folder in available_num_folders:
            num_path = os.path.join(main_path, num_folder)
            rcnet_folders = sorted([
                f for f in os.listdir(num_path)
                if os.path.isdir(os.path.join(num_path, f)) and f.startswith("RCNet")
            ])

            for rcnet_folder in rcnet_folders:
                rcnet_path = os.path.join(num_path, rcnet_folder)
                train_py_path = os.path.join(rcnet_path, "train.py")
                if os.path.exists(train_py_path):
                    info = {
                        'main_folder': main_folder,
                        'num_folder': num_folder,
                        'rcnet_folder': rcnet_folder,
                        'full_path': rcnet_path
                    }
                    train_scripts.append((train_py_path, info))
    return train_scripts


def run_single_training(train_info, python_cmd="python", timeout=None, gpu_manager=None):
    """Run single training script"""
    train_py_path, info = train_info
    work_dir = info['full_path']
    train_id = f"{info['main_folder']}/{info['num_folder']}/{info['rcnet_folder']}"
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training: {train_id} (Working directory: {work_dir})")

    gpu_id = gpu_manager.acquire_gpu(timeout=60) if gpu_manager else None

    start_time = time.time()
    error_info = ""
    try:
        env = os.environ.copy()
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Allocated GPU {gpu_id} for task {train_id}")
        else:
            env['CUDA_VISIBLE_DEVICES'] = '-1'
            logger.info(f"Task {train_id} running on CPU")

        result = subprocess.run(
            [python_cmd, "train.py"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
            encoding='utf-8'
        )

        status = "Success" if result.returncode == 0 else "Failed"
        if status == "Failed":
            logger.error(f"Training failed: {train_id} (Return code: {result.returncode})")
            logger.error(f"Error output: {result.stderr}")

    except subprocess.TimeoutExpired as e:
        status = "Timeout"
        logger.error(f"Training timeout: {train_id} (Timeout: {timeout}s)")
        result = None
        error_info = str(e)
    except Exception as e:
        status = "Exception"
        logger.error(f"Training exception: {train_id} - {str(e)}")
        result = None
        error_info = str(e)

    duration = time.time() - start_time
    if gpu_manager and gpu_id is not None:
        gpu_manager.release_gpu(gpu_id)

    return {
        'train_id': train_id,
        'status': status,
        'duration': duration,
        'returncode': result.returncode if result else -1,
        'stdout': result.stdout if result else "",
        'stderr': result.stderr if result else error_info,
        'work_dir': work_dir,
        'gpu_id': gpu_id
    }


def run_batch_training_parallel(train_scripts, max_workers=4, **kwargs):
    """Run multiple training scripts in parallel"""
    logger = logging.getLogger(__name__)
    gpu_manager = kwargs.get('gpu_manager')
    if gpu_manager and len(gpu_manager.available_gpus) > 0:
        max_workers = min(max_workers, len(gpu_manager.available_gpus))
        logger.info(f"Detected {len(gpu_manager.available_gpus)} GPUs, adjusting parallel count to {max_workers}")

    logger.info(f"Starting parallel training - Total tasks: {len(train_scripts)}, Parallel count: {max_workers}")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {executor.submit(run_single_training, ts, **kwargs): ts[1] for ts in train_scripts}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_info), 1):
            try:
                result = future.result()
                results.append(result)
                status_symbol = "✓" if result['status'] == 'Success' else "✗"
                gpu_info = f"GPU{result.get('gpu_id')}" if result.get('gpu_id') is not None else "CPU"
                logger.info(
                    f"{status_symbol} [{i}/{len(train_scripts)}] Task completed: {result['train_id']} - "
                    f"{result['status']} ({result['duration']:.1f}s, {gpu_info})"
                )
            except Exception as e:
                info = future_to_info[future]
                train_id = f"{info['main_folder']}/{info['num_folder']}/{info['rcnet_folder']}"
                logger.error(f"✗ [{i}/{len(train_scripts)}] Task execution exception: {train_id} - {e}")
                results.append({
                    'train_id': train_id,
                    'status': "Execution Exception",
                    'duration': 0,
                    'stderr': str(e)
                })

    return results


def run_batch_training_serial(train_scripts, **kwargs):
    """Run multiple training scripts serially"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting serial training - Total tasks: {len(train_scripts)}")
    results = [run_single_training(ts, **kwargs) for ts in train_scripts]
    return results


def save_results_summary(results, output_file):
    """Save results summary"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RCNet Batch Training Results Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        total = len(results)
        if total == 0:
            f.write("No tasks were executed.\n")
            return

        success = len([r for r in results if r['status'] == 'Success'])
        failed = len([r for r in results if r['status'] == 'Failed'])
        timeout = len([r for r in results if r['status'] == 'Timeout'])
        error = len([r for r in results if r['status'] in ['Exception', 'Execution Exception']])

        f.write("Overall Statistics:\n")
        f.write(f"Total tasks: {total}\n")
        f.write(f"  Success: {success} ({success/total*100:.1f}%)\n")
        f.write(f"  Failed: {failed} ({failed/total*100:.1f}%)\n")
        f.write(f"  Timeout: {timeout} ({timeout/total*100:.1f}%)\n")
        f.write(f"  Exception: {error} ({error/total*100:.1f}%)\n\n")

        f.write("Detailed Results:\n" + "-" * 50 + "\n")
        for result in sorted(results, key=lambda x: x['train_id']):
            f.write(f"Task: {result['train_id']}\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Duration: {result['duration']:.1f}s\n")
            f.write(f"  GPU: {result.get('gpu_id', 'CPU')}\n")
            if result.get('stderr'):
                f.write(f"  Error info: {str(result['stderr'])[:500].replace('%', '%%')}...\n")
            f.write("-" * 30 + "\n")


def main():
    """Main execution flow"""
    parser = argparse.ArgumentParser(
        description='Batch RCNet training script',
        epilog='''
Usage Instructions:
1. By default, the script intelligently recommends running mode based on GPU count and task count.
2. Multi-GPU + Multi-task → Recommended parallel mode.
3. Single GPU or Single task → Automatically use serial mode.

Supported datasets:
- 124: space_group + pearson_symbol + c_a_ratio
- 125: space_group + pearson_symbol + beta_angle
- 12345: all feature combinations

Frequency length range: 6-78 (step 6)
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--rcnet-training-dir',
                        default='./rcnet_training_umap',
                        help='RCNet training root directory')
    parser.add_argument('--main-folders', nargs='+', help='Specify main folders (e.g.: 124 125)')
    parser.add_argument('--num-folders', nargs='+', help='Specify number folders (e.g.: 6 12 18)')
    parser.add_argument('--parallel', action='store_true', help='Force parallel execution')
    parser.add_argument('--serial', action='store_true', help='Force serial execution')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum parallel workers')
    parser.add_argument('--python-cmd', default='python', help='Python command')
    parser.add_argument('--timeout', type=int, help='Timeout for single training (seconds)')
    parser.add_argument('--log-dir', default='./batch_logs/', help='Log directory')
    parser.add_argument('--dry-run', action='store_true', help='Only list scripts, do not execute')
    parser.add_argument('--gpu-ids', nargs='+', type=int, help='Specify GPU IDs to use')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU mode')
    parser.add_argument('--yes', '-y', action='store_true', help='Auto-confirm all interactive prompts')

    # Use parse_known_args() for Jupyter Notebook compatibility
    args, unknown = parser.parse_known_args()

    stdin_is_tty = sys.stdin.isatty()
    interactive = stdin_is_tty and not args.yes
    auto_confirm = args.yes or not stdin_is_tty

    logger = setup_logging(args.log_dir)

    logger.info("=" * 60)
    logger.info("RCNet Batch Training Script")
    logger.info(f"RCNet training directory: {args.rcnet_training_dir}")
    logger.info("=" * 60)

    gpu_manager = None
    if not args.cpu_only:
        gpu_manager = GPUManager(available_gpus=args.gpu_ids)
        if not gpu_manager.available_gpus:
            gpu_manager = None
    else:
        logger.info("Force CPU mode")

    logger.info("Searching for training scripts...")
    train_scripts = find_train_scripts(args.rcnet_training_dir, args.main_folders, args.num_folders)

    if not train_scripts:
        logger.error("No training scripts found! Please check paths and filter conditions.")
        return

    logger.info(f"Found {len(train_scripts)} training scripts")
    if args.dry_run:
        for script_path, info in train_scripts:
            logger.info(f"  - {info['main_folder']}/{info['num_folder']}/{info['rcnet_folder']}")
        logger.info("Dry run mode, exiting.")
        return

    parallel_mode = False
    if args.parallel:
        parallel_mode = True
    elif not args.serial:
        gpu_count = len(gpu_manager.available_gpus) if gpu_manager else 0
        if gpu_count > 1 and len(train_scripts) > 1:
            if interactive:
                try:
                    mode_choice = input(
                        f"Detected {gpu_count} GPUs and {len(train_scripts)} tasks, run in parallel? (Y/n): "
                    ).strip().lower()
                    parallel_mode = mode_choice != 'n'
                except (KeyboardInterrupt, EOFError):
                    logger.info("User cancelled, exiting.")
                    return
            else:
                parallel_mode = True
                logger.info("Auto-selected parallel mode (non-interactive run).")
    mode_text = "parallel" if parallel_mode else "serial"
    logger.info(f"Selected running mode: {mode_text}")

    if auto_confirm:
        logger.info("Auto-confirmed execution.")
    else:
        try:
            confirm = input(
                f"Confirm to run these {len(train_scripts)} training tasks in {mode_text} mode? (y/N): "
            ).lower()
            if confirm not in ['y', 'yes']:
                logger.info("User cancelled execution.")
                return
        except (KeyboardInterrupt, EOFError):
            logger.info("User cancelled, exiting.")
            return

    logger.info(f"Starting {mode_text} training...")
    start_time = time.time()

    run_kwargs = {'python_cmd': args.python_cmd, 'timeout': args.timeout, 'gpu_manager': gpu_manager}
    if parallel_mode:
        results = run_batch_training_parallel(train_scripts, args.max_workers, **run_kwargs)
    else:
        results = run_batch_training_serial(train_scripts, **run_kwargs)

    total_duration = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info("Batch training completed!")
    logger.info(f"Total duration: {total_duration/60:.1f} minutes ({total_duration:.0f} seconds)")
    summary_file = os.path.join(args.log_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    save_results_summary(results, summary_file)
    logger.info(f"Results summary saved to: {summary_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
