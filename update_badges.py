import re

def get_pylint_score(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    match = re.search(r'Your code has been rated at ([\d\.]+)/10', content)
    return match.group(1) if match else "N/A"

def get_coverage(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    match = re.search(r'TOTAL\s+\d+\s+\d+\s+\d+\s+(\d+%)', content)
    return match.group(1) if match else "N/A"

def update_readme(pylint_score):
    with open("README.md", "r") as file:
        readme = file.read()

    readme = re.sub(r'!\[Pylint\]\([^)]+\)', f'![Pylint](https://img.shields.io/badge/pylint-{pylint_score}-green)', readme)
    #readme = re.sub(r'!\[Coverage\]\([^)]+\)', f'![Coverage](https://img.shields.io/badge/coverage-{coverage_score}-brightgreen)', readme)

    with open("README.md", "w") as file:
        file.write(readme)

if __name__ == "__main__":
    pylint_score = get_pylint_score("pylint_report.txt")
    #coverage_score = get_coverage("coverage.txt")
    update_readme(pylint_score)