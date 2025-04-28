#!/usr/bin/env python3
"""
Script to run tests for the RNA 3D folding project.
"""

import os
import sys
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for the RNA 3D folding project")

    parser.add_argument(
        "--package",
        action="store_true",
        help="Run package structure tests"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run data download tests"
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Run visualization tests"
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run data analysis tests"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )

    return parser.parse_args()

def main():
    """Main function to run tests."""
    args = parse_args()

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                 RNA 3D Structure Test Suite                     ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print("")

    # If no arguments provided, run all tests
    if not (args.package or args.download or args.visualization or args.analysis or args.all):
        args.all = True
        print("▶ No specific test category selected, running all tests")
    else:
        print("▶ Running selected test categories:")
        if args.all:
            print("  • All tests")
        else:
            if args.package:
                print("  • Package structure tests")
            if args.download:
                print("  • Data download tests")
            if args.visualization:
                print("  • Visualization tests")
            if args.analysis:
                print("  • Data analysis tests")

    print("")

    # Determine which tests to run
    if args.all:
        cmd = "pytest tests/ -v"
    else:
        test_paths = []

        if args.package:
            test_paths.append("tests/test_package.py")

        if args.download:
            test_paths.append("tests/data/test_download.py")

        if args.visualization:
            test_paths.append("tests/visualization/test_visualization.py")

        if args.analysis:
            test_paths.append("tests/data/test_analysis.py")

        cmd = f"pytest {' '.join(test_paths)} -v"

    # Run tests
    print(f"Running command: {cmd}")
    print("")
    result = os.system(cmd)

    print("")
    if result == 0:
        print("✓ All tests passed successfully!")
        print("  The codebase is in a green state and ready for use.")
    else:
        print("⚠ Some tests failed.")
        print("  Please review the test output above and fix any issues before proceeding.")

    print("")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                        Test Run Complete                        ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    return result

if __name__ == "__main__":
    sys.exit(main())
