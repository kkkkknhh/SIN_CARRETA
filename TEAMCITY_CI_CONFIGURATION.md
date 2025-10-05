# TeamCity CI Configuration for Python Projects

This document provides comprehensive instructions for configuring TeamCity CI with virtual environments, environment isolation, and pip caching for optimal Python project builds.

## Overview

This configuration ensures:
- ✅ Isolated Python virtual environment per build
- ✅ Prevention of user site-packages interference
- ✅ Pip caching for improved build performance
- ✅ Reproducible builds across different agents

## Prerequisites

- TeamCity Server (2021.1 or later recommended)
- Build agents with Python 3.7+ installed
- Agent work folder with write permissions

## Build Agent Configuration

### 1. Agent Properties Configuration

Add the following properties to your build agent configuration (`buildAgent.properties`):

```properties
# Python virtual environment configuration
python.virtualenv.path=%agent.work.dir%/.venv
python.interpreter.path=%agent.work.dir%/.venv/bin/python
pip.cache.dir=%agent.work.dir%/.pip-cache

# System capabilities
system.python3=/usr/bin/python3
system.pip3=/usr/bin/pip3
```

### 2. Agent Environment Variables

Configure these environment variables on your build agents:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONNOUSERSITE` | `1` | Prevents user site-packages from interfering |
| `PIP_CACHE_DIR` | `%agent.work.dir%/.pip-cache` | Centralized pip cache location |
| `VIRTUAL_ENV` | `%agent.work.dir%/.venv` | Virtual environment path |

## Project Configuration

### Build Configuration Template

Create a new build configuration with the following steps:

#### Step 1: Setup Virtual Environment
```xml
<build-step id="setup_venv" type="simpleRunner">
    <parameters>
        <param name="script.content"><![CDATA[
#!/bin/bash
set -e

# Remove existing venv if it exists
rm -rf $(Agent.WorkFolder)/.venv

# Create new virtual environment
python3 -m venv $(Agent.WorkFolder)/.venv

# Activate virtual environment and upgrade pip
source $(Agent.WorkFolder)/.venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "Virtual environment created at: $(Agent.WorkFolder)/.venv"
$(Agent.WorkFolder)/.venv/bin/python --version
        ]]></param>
        <param name="teamcity.step.mode">default</param>
        <param name="use.custom.script">true</param>
    </parameters>
</build-step>
```

#### Step 2: Install Dependencies
```xml
<build-step id="install_deps" type="simpleRunner">
    <parameters>
        <param name="script.content"><![CDATA[
#!/bin/bash
set -e

# Use virtual environment python
PYTHON_EXEC="$(Agent.WorkFolder)/.venv/bin/python"
PIP_EXEC="$(Agent.WorkFolder)/.venv/bin/pip"

# Install project dependencies with caching
$PIP_EXEC install --cache-dir $(Agent.WorkFolder)/.pip-cache -r requirements.txt

# Verify installation
$PYTHON_EXEC -c "import sys; print('Python path:', sys.executable)"
$PIP_EXEC list
        ]]></param>
        <param name="teamcity.step.mode">default</param>
        <param name="use.custom.script">true</param>
    </parameters>
</build-step>
```

#### Step 3: Build Validation
```xml
<build-step id="build_check" type="simpleRunner">
    <parameters>
        <param name="script.content"><![CDATA[
#!/bin/bash
set -e

PYTHON_EXEC="$(Agent.WorkFolder)/.venv/bin/python"

# Validate build
$PYTHON_EXEC -c "import feasibility_scorer; print('Build successful')"

# Show Python environment info
echo "=== Python Environment Info ==="
$PYTHON_EXEC -c "
import sys, os
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')
print(f'Python path: {sys.path[:3]}')
print(f'PYTHONNOUSERSITE: {os.environ.get(\"PYTHONNOUSERSITE\", \"Not set\")}')
print(f'Virtual env: {os.environ.get(\"VIRTUAL_ENV\", \"Not set\")}')
"
        ]]></param>
        <param name="teamcity.step.mode">default</param>
        <param name="use.custom.script">true</param>
    </parameters>
</build-step>
```

#### Step 4: Run Linting
```xml
<build-step id="lint_check" type="simpleRunner">
    <parameters>
        <param name="script.content"><![CDATA[
#!/bin/bash
set -e

PYTHON_EXEC="$(Agent.WorkFolder)/.venv/bin/python"

# Run lint check
$PYTHON_EXEC -c "import py_compile; py_compile.compile('feasibility_scorer.py', doraise=True); print('Lint successful')"
        ]]></param>
        <param name="teamcity.step.mode">default</param>
        <param name="use.custom.script">true</param>
    </parameters>
</build-step>
```

#### Step 5: Run Tests
```xml
<build-step id="run_tests" type="simpleRunner">
    <parameters>
        <param name="script.content"><![CDATA[
#!/bin/bash
set -e

PYTHON_EXEC="$(Agent.WorkFolder)/.venv/bin/python"

# Run tests with pytest (if available) or fallback to custom runner
if $PYTHON_EXEC -m pytest --version > /dev/null 2>&1; then
    echo "Running tests with pytest..."
    $PYTHON_EXEC -m pytest test_feasibility_scorer.py -v --tb=short
else
    echo "Running tests with custom runner..."
    $PYTHON_EXEC run_tests.py
fi
        ]]></param>
        <param name="teamcity.step.mode">default</param>
        <param name="use.custom.script">true</param>
    </parameters>
</build-step>
```

### Environment Variables Configuration

Configure the following environment variables in your build configuration:

```xml
<parameters>
    <param name="env.PYTHONNOUSERSITE" value="1" />
    <param name="env.PIP_CACHE_DIR" value="%agent.work.dir%/.pip-cache" />
    <param name="env.VIRTUAL_ENV" value="%agent.work.dir%/.venv" />
    <param name="env.PATH" value="%agent.work.dir%/.venv/bin:%env.PATH%" />
</parameters>
```

## Sample Build Configuration (XML)

Here's a complete build configuration that you can import into TeamCity:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<build-type xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
            xsi:noNamespaceSchemaLocation="https://www.jetbrains.com/teamcity/schemas/2021.1/project-config.xsd" 
            uuid="FeasibilityScorer_Build" 
            id="FeasibilityScorer_Build" 
            name="Feasibility Scorer - Python CI">
    
    <description>Python CI pipeline with virtual environment and pip caching</description>
    
    <settings>
        <parameters>
            <param name="env.PYTHONNOUSERSITE" value="1" />
            <param name="env.PIP_CACHE_DIR" value="%agent.work.dir%/.pip-cache" />
            <param name="env.VIRTUAL_ENV" value="%agent.work.dir%/.venv" />
            <param name="env.PATH" value="%agent.work.dir%/.venv/bin:%env.PATH%" />
        </parameters>
        
        <build-runners>
            <!-- Setup Virtual Environment -->
            <runner id="setup_venv" name="Setup Virtual Environment" type="simpleRunner">
                <parameters>
                    <param name="script.content"><![CDATA[
#!/bin/bash
set -e

echo "=== Setting up Python Virtual Environment ==="
echo "Agent work folder: %agent.work.dir%"
echo "Python version: $(python3 --version)"

# Remove existing venv if it exists  
rm -rf %agent.work.dir%/.venv

# Create new virtual environment
python3 -m venv %agent.work.dir%/.venv

# Activate and upgrade pip
source %agent.work.dir%/.venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "Virtual environment created successfully at: %agent.work.dir%/.venv"
%agent.work.dir%/.venv/bin/python --version
                    ]]></param>
                    <param name="teamcity.step.mode">default</param>
                    <param name="use.custom.script">true</param>
                </parameters>
            </runner>
            
            <!-- Install Dependencies -->
            <runner id="install_deps" name="Install Dependencies" type="simpleRunner">
                <parameters>
                    <param name="script.content"><![CDATA[
#!/bin/bash
set -e

echo "=== Installing Dependencies ==="
PYTHON_EXEC="%agent.work.dir%/.venv/bin/python"
PIP_EXEC="%agent.work.dir%/.venv/bin/pip"

# Create pip cache directory if it doesn't exist
mkdir -p %agent.work.dir%/.pip-cache

# Install dependencies with caching
$PIP_EXEC install --cache-dir %agent.work.dir%/.pip-cache -r requirements.txt

echo "=== Installed Packages ==="
$PIP_EXEC list

echo "=== Python Environment Verification ==="
$PYTHON_EXEC -c "
import sys, os
print(f'Python executable: {sys.executable}')
print(f'PYTHONNOUSERSITE: {os.environ.get(\"PYTHONNOUSERSITE\", \"Not set\")}')
print(f'Virtual env active: {hasattr(sys, \"real_prefix\") or (hasattr(sys, \"base_prefix\") and sys.base_prefix != sys.prefix)}')
"
                    ]]></param>
                    <param name="teamcity.step.mode">default</param>
                    <param name="use.custom.script">true</param>
                </parameters>
            </runner>
            
            <!-- Build Check -->
            <runner id="build_check" name="Build Validation" type="simpleRunner">
                <parameters>
                    <param name="script.content"><![CDATA[
#!/bin/bash
set -e

echo "=== Build Validation ==="
PYTHON_EXEC="%agent.work.dir%/.venv/bin/python"

# Validate main module imports
$PYTHON_EXEC -c "import feasibility_scorer; print('✓ Main module imports successfully')"

# Validate module functionality
$PYTHON_EXEC -c "
from feasibility_scorer import FeasibilityScorer
scorer = FeasibilityScorer()
result = scorer.calculate_feasibility_score('línea base 50% meta 80% año 2025')
print(f'✓ Module functional test passed: score={result.feasibility_score}')
"

echo "Build validation completed successfully!"
                    ]]></param>
                    <param name="teamcity.step.mode">default</param>
                    <param name="use.custom.script">true</param>
                </parameters>
            </runner>
            
            <!-- Lint Check -->
            <runner id="lint_check" name="Code Linting" type="simpleRunner">
                <parameters>
                    <param name="script.content"><![CDATA[
#!/bin/bash
set -e

echo "=== Code Linting ==="
PYTHON_EXEC="%agent.work.dir%/.venv/bin/python"

# Compile check for syntax errors
$PYTHON_EXEC -c "import py_compile; py_compile.compile('feasibility_scorer.py', doraise=True); print('✓ Lint successful - no syntax errors')"

echo "Linting completed successfully!"
                    ]]></param>
                    <param name="teamcity.step.mode">default</param>
                    <param name="use.custom.script">true</param>
                </parameters>
            </runner>
            
            <!-- Run Tests -->
            <runner id="run_tests" name="Run Tests" type="simpleRunner">
                <parameters>
                    <param name="script.content"><![CDATA[
#!/bin/bash
set -e

echo "=== Running Tests ==="
PYTHON_EXEC="%agent.work.dir%/.venv/bin/python"

# Try pytest first, fallback to custom runner
if $PYTHON_EXEC -m pytest --version > /dev/null 2>&1; then
    echo "Running tests with pytest..."
    $PYTHON_EXEC -m pytest test_feasibility_scorer.py -v --tb=short
else
    echo "Running tests with custom runner..."
    $PYTHON_EXEC run_tests.py
fi

echo "All tests completed successfully!"
                    ]]></param>
                    <param name="teamcity.step.mode">default</param>
                    <param name="use.custom.script">true</param>
                </parameters>
            </runner>
        </build-runners>
        
        <vcs-settings>
            <vcs-entry-ref root-id="FeasibilityScorer_GitRoot" />
        </vcs-settings>
        
        <cleanup />
    </settings>
    
    <requirements>
        <equals name="system.python3" value="*" />
    </requirements>
</build-type>
```

## Performance Optimizations

### Pip Cache Configuration

The pip cache significantly improves build times by reusing downloaded packages:

```bash
# Cache location per agent
PIP_CACHE_DIR=%agent.work.dir%/.pip-cache

# Cache size monitoring
du -sh %agent.work.dir%/.pip-cache
```

### Virtual Environment Reuse (Optional)

For faster builds, you can optionally reuse virtual environments between builds:

```bash
# Check if venv exists and is valid
if [ -f "%agent.work.dir%/.venv/bin/python" ] && [ -f "%agent.work.dir%/.venv/pyvenv.cfg" ]; then
    echo "Reusing existing virtual environment"
    source %agent.work.dir%/.venv/bin/activate
    pip install --upgrade -r requirements.txt
else
    echo "Creating new virtual environment"
    # ... standard venv creation
fi
```

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure agent work folder has write permissions
2. **Python not found**: Verify Python 3.7+ is installed on build agents
3. **Module import errors**: Check `PYTHONNOUSERSITE=1` is set properly
4. **Cache issues**: Clear pip cache with `rm -rf %agent.work.dir%/.pip-cache`

### Verification Commands

```bash
# Check virtual environment
ls -la %agent.work.dir%/.venv/bin/

# Verify Python isolation  
%agent.work.dir%/.venv/bin/python -c "import sys; print(sys.path)"

# Check environment variables
printenv | grep PYTHON
```

## Testing the Configuration

### Manual Testing Steps

1. **Create Test Build**: Import the sample configuration above
2. **Run Pipeline**: Trigger a build manually
3. **Verify Logs**: Check each step completes successfully
4. **Validate Environment**: Confirm virtual environment isolation
5. **Performance Check**: Monitor pip cache utilization

### Expected Results

✅ Virtual environment created at `%agent.work.dir%/.venv`  
✅ Dependencies installed with pip caching  
✅ Build validation passes  
✅ Linting completes without errors  
✅ All tests pass  
✅ No user site-packages interference  

## Maintenance

### Regular Tasks

- **Weekly**: Monitor pip cache size and clean if >1GB
- **Monthly**: Update base Python image on build agents  
- **Per Release**: Verify dependency compatibility

### Cache Management

```bash
# Clean pip cache (run monthly)
rm -rf %agent.work.dir%/.pip-cache/*

# Clean old virtual environments  
find %agent.work.dir% -name ".venv*" -type d -mtime +7 -exec rm -rf {} \;
```

---

This configuration ensures reliable, performant Python CI builds with proper environment isolation and caching optimization.