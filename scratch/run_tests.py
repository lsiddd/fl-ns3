#!/usr/bin/env python3
"""
Test runner script for FL API tests.
Provides convenient ways to run different test suites.
"""

import subprocess
import sys
import argparse
import os


def run_command(cmd, description):
    """Run a command and handle the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install -r test_requirements.txt")
        return False


def install_test_dependencies():
    """Install test dependencies"""
    return run_command(
        [sys.executable, "-m", "pip", "install", "-r", "test_requirements.txt"],
        "Installing test dependencies"
    )


def run_unit_tests():
    """Run unit tests only"""
    return run_command(
        ["python", "-m", "pytest", "-v", "-m", "not integration", "--tb=short"],
        "Unit tests"
    )


def run_integration_tests():
    """Run integration tests only"""
    return run_command(
        ["python", "-m", "pytest", "-v", "-m", "integration", "--tb=short"],
        "Integration tests"
    )


def run_all_tests():
    """Run all tests"""
    return run_command(
        ["python", "-m", "pytest", "-v", "--tb=short"],
        "All tests"
    )


def run_fast_tests():
    """Run fast tests only (exclude slow tests)"""
    return run_command(
        ["python", "-m", "pytest", "-v", "-m", "not slow", "--tb=short"],
        "Fast tests (excluding slow dataset tests)"
    )


def run_coverage_tests():
    """Run tests with detailed coverage report"""
    return run_command(
        ["python", "-m", "pytest", "-v", "--cov=.", "--cov-report=term-missing", "--cov-report=html"],
        "Tests with coverage report"
    )


def run_specific_test(test_pattern):
    """Run specific test based on pattern"""
    return run_command(
        ["python", "-m", "pytest", "-v", "-k", test_pattern, "--tb=short"],
        f"Tests matching pattern: {test_pattern}"
    )


def main():
    parser = argparse.ArgumentParser(description="FL API Test Runner")
    parser.add_argument(
        "command",
        choices=["install", "unit", "integration", "all", "fast", "coverage", "specific"],
        help="Test command to run"
    )
    parser.add_argument(
        "--pattern", "-p",
        help="Test pattern for 'specific' command (e.g., 'test_config' or 'TestFLConfig')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = True
    
    if args.command == "install":
        success = install_test_dependencies()
    elif args.command == "unit":
        success = run_unit_tests()
    elif args.command == "integration":
        success = run_integration_tests()
    elif args.command == "all":
        success = run_all_tests()
    elif args.command == "fast":
        success = run_fast_tests()
    elif args.command == "coverage":
        success = run_coverage_tests()
    elif args.command == "specific":
        if not args.pattern:
            print("❌ --pattern is required for 'specific' command")
            sys.exit(1)
        success = run_specific_test(args.pattern)
    
    if success:
        print(f"\n🎉 Test command '{args.command}' completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Test command '{args.command}' failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()