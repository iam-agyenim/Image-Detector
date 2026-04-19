#!/usr/bin/env python
"""Unified CLI entry point for the Image Detector toolkit.

Usage:
    python cli.py detect    [options]   – single-image object detection
    python cli.py batch     [options]   – batch detection on a directory
    python cli.py report    [options]   – generate a JSON detection report
    python cli.py compare   [options]   – compare objects between two images
    python cli.py stats     [options]   – aggregate statistics from a report
"""

import sys


COMMANDS = {
    "detect": "detection",
    "batch": "batch_detection",
    "report": "report",
    "compare": "compare",
    "stats": "stats",
}


def usage():
    print("Usage: python cli.py <command> [options]\n")
    print("Available commands:")
    print("  detect   – Single-image object detection")
    print("  batch    – Batch detection on a directory of images")
    print("  report   – Generate a JSON detection report")
    print("  compare  – Compare detected objects between two images")
    print("  stats    – Aggregate statistics from a JSON report")
    print("\nRun 'python cli.py <command> --help' for command-specific options.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        usage()
        sys.exit(0)

    command = sys.argv[1]

    if command not in COMMANDS:
        print(f"Error: Unknown command '{command}'.\n")
        usage()
        sys.exit(1)

    module_name = COMMANDS[command]
    sys.argv = sys.argv[1:]
    sys.argv[0] = f"cli.py {command}"

    module = __import__(module_name)
    module.main()


if __name__ == "__main__":
    main()
