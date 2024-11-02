from typing import List, Dict
import subprocess

dependencies_list = {
    'python': ['llama-index', 'transformers', 'PyMuPDF', 'groq'],
}

def setup_tools(dependencies):
    for category, deps in dependencies_list.items():
        for dependency in deps:
            if dependency:
                print(f'Installing dependency: {dependency}')
                try:
                    # Install each dependency using subprocess
                    subprocess.run(['pip', 'install', '-q', dependency], check=True)
                except subprocess.CalledProcessError as e:
                    print(f'Failed to install {dependency}: {e}')

setup_tools(dependencies_list)