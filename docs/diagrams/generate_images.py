#!/usr/bin/env python3
"""Generate PNG images from DOT files using Graphviz."""

import os
import glob
import subprocess
import sys

def generate_images():
    """Generate PNG images from all DOT files in current directory."""
    diagrams_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(diagrams_dir)
    
    dot_files = glob.glob("*.dot")
    
    if not dot_files:
        print("No .dot files found")
        return 1
    
    print(f"Found {len(dot_files)} DOT files")
    
    success_count = 0
    failed_files = []
    
    for dot_file in sorted(dot_files):
        png_file = dot_file.replace('.dot', '.png')
        print(f"Generating {png_file}...")
        
        try:
            # Try using dot command
            result = subprocess.run(
                ['dot', '-Tpng', '-Gdpi=300', dot_file, '-o', png_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                file_size = os.path.getsize(png_file)
                print(f"  ✓ Generated {png_file} ({file_size} bytes)")
                success_count += 1
            else:
                print(f"  ✗ Failed: {result.stderr}")
                failed_files.append(dot_file)
                
        except FileNotFoundError:
            # Try using Python graphviz library
            try:
                from graphviz import Source
                source = Source.from_file(dot_file)
                source.render(filename=dot_file.replace('.dot', ''), format='png', cleanup=True)
                file_size = os.path.getsize(png_file)
                print(f"  ✓ Generated {png_file} using Python graphviz ({file_size} bytes)")
                success_count += 1
            except Exception as e:
                print(f"  ✗ Failed with Python graphviz: {e}")
                failed_files.append(dot_file)
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_files.append(dot_file)
    
    print(f"\n{'='*60}")
    print(f"Generated {success_count}/{len(dot_files)} images successfully")
    
    if failed_files:
        print(f"Failed files: {', '.join(failed_files)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(generate_images())
