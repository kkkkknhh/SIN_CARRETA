#!/usr/bin/env python3
"""
Test script to validate TeamCity CI setup and virtual environment configuration.
This script simulates the TeamCity build process to verify all components work correctly.
"""

import os
import shutil
import subprocess
import sys
import tempfile


class TeamCitySetupTester:
    """Tests the TeamCity CI configuration steps."""

    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.venv_path = None

    def log_test(self, test_name, passed, message=""):
        """Log test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        result = f"{status}: {test_name}"
        if message:
            result += f" - {message}"
        print(result)
        self.test_results.append((test_name, passed, message))

    def setup_test_environment(self):
        """Create temporary environment for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="teamcity_test_")
        self.venv_path = os.path.join(self.temp_dir, ".venv")
        print(f"Test environment: {self.temp_dir}")

    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up: {self.temp_dir}")

    def test_python_availability(self):
        """Test Python 3 availability."""
        try:
            result = subprocess.run(
                ["python3", "--version"], capture_output=True, text=True, timeout=10, 
            check=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log_test("Python3 Available", True, version)
                return True
            else:
                self.log_test("Python3 Available", False,
                              "python3 command failed")
                return False
        except Exception as e:
            self.log_test("Python3 Available", False, str(e))
            return False

    def test_virtual_environment_creation(self):
        """Test virtual environment creation."""
        try:
            # Create virtual environment
            cmd = ["python3", "-m", "venv", self.venv_path]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=True)

            if result.returncode == 0 and os.path.exists(self.venv_path):
                # Check if python executable exists
                python_exec = os.path.join(self.venv_path, "bin", "python")
                if os.path.exists(python_exec):
                    self.log_test(
                        "Virtual Environment Creation",
                        True,
                        f"Created at {self.venv_path}",
                    )
                    return True
                else:
                    self.log_test(
                        "Virtual Environment Creation",
                        False,
                        "Python executable not found",
                    )
                    return False
            else:
                self.log_test("Virtual Environment Creation",
                              False, result.stderr)
                return False
        except Exception as e:
            self.log_test("Virtual Environment Creation", False, str(e))
            return False

    def test_environment_isolation(self):
        """Test environment isolation with PYTHONNOUSERSITE."""
        try:
            python_exec = os.path.join(self.venv_path, "bin", "python")
            if not os.path.exists(python_exec):
                self.log_test(
                    "Environment Isolation", False, "Virtual environment not found"
                )
                return False

            # Test with PYTHONNOUSERSITE=1
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"

            cmd = [
                python_exec,
                "-c",
                "import sys, os; print(f'PYTHONNOUSERSITE={os.environ.get(\"PYTHONNOUSERSITE\", \"Not set\")}'); print(f'UserSite: {sys.flags.no_user_site}')",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=10, 
            check=True)

            if result.returncode == 0:
                output = result.stdout.strip()
                if "PYTHONNOUSERSITE=1" in output and "UserSite: True" in output:
                    self.log_test(
                        "Environment Isolation", True, "PYTHONNOUSERSITE working"
                    )
                    return True
                else:
                    self.log_test(
                        "Environment Isolation", False, f"Unexpected output: {output}"
                    )
                    return False
            else:
                self.log_test("Environment Isolation", False, result.stderr)
                return False
        except Exception as e:
            self.log_test("Environment Isolation", False, str(e))
            return False

    def test_pip_caching(self):
        """Test pip caching functionality."""
        try:
            python_exec = os.path.join(self.venv_path, "bin", "python")
            pip_exec = os.path.join(self.venv_path, "bin", "pip")
            cache_dir = os.path.join(self.temp_dir, ".pip-cache")

            if not os.path.exists(python_exec):
                self.log_test("Pip Caching", False,
                              "Virtual environment not found")
                return False

            # Create cache directory
            os.makedirs(cache_dir, exist_ok=True)

            # Test pip with cache directory
            env = os.environ.copy()
            env["PIP_CACHE_DIR"] = cache_dir

            cmd = [
                pip_exec,
                "install",
                "--cache-dir",
                cache_dir,
                "--dry-run",
                "setuptools",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=30, 
            check=True)

            if result.returncode == 0:
                self.log_test("Pip Caching", True,
                              f"Cache directory: {cache_dir}")
                return True
            else:
                self.log_test("Pip Caching", False, result.stderr)
                return False
        except Exception as e:
            self.log_test("Pip Caching", False, str(e))
            return False

    def test_project_dependencies(self):
        """Test installation of project dependencies."""
        try:
            # Copy requirements.txt to test directory
            current_dir = os.getcwd()
            requirements_file = os.path.join(current_dir, "requirements.txt")

            if not os.path.exists(requirements_file):
                self.log_test(
                    "Project Dependencies", False, "requirements.txt not found"
                )
                return False

            test_requirements = os.path.join(self.temp_dir, "requirements.txt")
            shutil.copy2(requirements_file, test_requirements)

            # Install dependencies
            pip_exec = os.path.join(self.venv_path, "bin", "pip")
            cache_dir = os.path.join(self.temp_dir, ".pip-cache")

            cmd = [
                pip_exec,
                "install",
                "--cache-dir",
                cache_dir,
                "-r",
                test_requirements,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=True)

            if result.returncode == 0:
                self.log_test(
                    "Project Dependencies", True, "Dependencies installed successfully"
                )
                return True
            else:
                self.log_test("Project Dependencies", False, result.stderr)
                return False
        except Exception as e:
            self.log_test("Project Dependencies", False, str(e))
            return False

    def test_module_import(self):
        """Test importing the main project module."""
        try:
            # Copy project files to test directory
            current_dir = os.getcwd()
            project_files = ["feasibility_scorer.py"]

            for file_name in project_files:
                src_file = os.path.join(current_dir, file_name)
                if os.path.exists(src_file):
                    dst_file = os.path.join(self.temp_dir, file_name)
                    shutil.copy2(src_file, dst_file)

            # Test import
            python_exec = os.path.join(self.venv_path, "bin", "python")

            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONPATH"] = self.temp_dir

            cmd = [
                python_exec,
                "-c",
                "import feasibility_scorer; print('Import successful')",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
                cwd=self.temp_dir,
            check=True)

            if result.returncode == 0 and "Import successful" in result.stdout:
                self.log_test(
                    "Module Import", True, "feasibility_scorer imported successfully"
                )
                return True
            else:
                self.log_test("Module Import", False,
                              f"Import failed: {result.stderr}")
                return False
        except Exception as e:
            self.log_test("Module Import", False, str(e))
            return False

    def test_build_validation(self):
        """Test build validation step."""
        try:
            python_exec = os.path.join(self.venv_path, "bin", "python")

            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONPATH"] = self.temp_dir

            # Test functionality
            cmd = [
                python_exec,
                "-c",
                """
from feasibility_scorer import FeasibilityScorer
scorer = FeasibilityScorer()
result = scorer.calculate_feasibility_score('línea base 50% meta 80% año 2025')
print(f'Build validation passed: score={result.feasibility_score}')
""",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
                cwd=self.temp_dir,
            check=True)

            if result.returncode == 0 and "Build validation passed" in result.stdout:
                self.log_test("Build Validation", True,
                              "Functional test passed")
                return True
            else:
                self.log_test(
                    "Build Validation", False, f"Validation failed: {result.stderr}"
                )
                return False
        except Exception as e:
            self.log_test("Build Validation", False, str(e))
            return False

    def test_lint_check(self):
        """Test code linting step."""
        try:
            python_exec = os.path.join(self.venv_path, "bin", "python")

            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONPATH"] = self.temp_dir

            cmd = [
                python_exec,
                "-c",
                """
import py_compile
py_compile.compile('feasibility_scorer.py', doraise=True)
print('Lint successful')
""",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
                cwd=self.temp_dir,
            check=True)

            if result.returncode == 0 and "Lint successful" in result.stdout:
                self.log_test("Lint Check", True, "No syntax errors found")
                return True
            else:
                self.log_test("Lint Check", False, result.stderr)
                return False
        except Exception as e:
            self.log_test("Lint Check", False, str(e))
            return False

    def run_all_tests(self):
        """Run all TeamCity setup tests."""
        print("=" * 60)
        print("TEAMCITY CI CONFIGURATION VALIDATION")
        print("=" * 60)

        try:
            self.setup_test_environment()

            # Run tests in order
            tests = [
                self.test_python_availability,
                self.test_virtual_environment_creation,
                self.test_environment_isolation,
                self.test_pip_caching,
                self.test_project_dependencies,
                self.test_module_import,
                self.test_build_validation,
                self.test_lint_check,
            ]

            passed_tests = 0
            for test in tests:
                if test():
                    passed_tests += 1

        finally:
            self.cleanup_test_environment()

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed = sum(1 for _, passed, _ in self.test_results if passed)
        failed = total_tests - passed

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFAILED TESTS:")
            for test_name, passed, message in self.test_results:
                if not passed:
                    print(f"  ✗ {test_name}: {message}")

        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")

        if success_rate >= 100:
            print(
                "\n🎉 All tests passed! TeamCity configuration is ready for deployment."
            )
            return 0
        elif success_rate >= 80:
            print(
                "\n⚠️  Most tests passed. Minor issues may need attention before deployment."
            )
            return 1
        else:
            print(
                "\n❌ Significant issues found. Review configuration before deployment."
            )
            return 2


def main():
    """Main entry point."""
    tester = TeamCitySetupTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
