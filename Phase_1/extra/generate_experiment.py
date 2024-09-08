import json
import logging
from typing import Dict, Any
from pydantic import BaseModel, ValidationError
import subprocess
import os
import os.path as osp
import importlib
from .model_implementations import ModelInterface

class ExperimentCodeOutput(BaseModel):
    experiment_code: str
    plot_code: str

def check_dependencies(code: str) -> bool:
    allowed_libraries = {
        'numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'seaborn',
        'torch', 'torchvision', 'torchaudio', 'xgboost', 'lightgbm', 'catboost',
        'statsmodels', 'plotly', 'PIL', 'cv2', 'nltk', 'gensim', 'spacy', 'joblib',
        'requests', 'deap', 'sympy', 'networkx', 'pytest', 'bokeh', 'dask',
        'pyspark', 'skimage', 'imblearn', 'warnings', 'os', 'sys', 'json', 'random', 'logging', 'time', 'datetime', 're', 'collections', 'itertools', 'math', 'copy', 'shutil', 'subprocess', 'multiprocessing', 'threading', 'concurrent', 'queue', 'asyncio', 'socket', 'http', 'urllib', 'ftplib', 'email', 'smtplib', 'csv', 'sqlite3', 'sqlalchemy', 'pickle', 'json', 'yaml', 'xml', 'html', 'zipfile', 'tarfile',
    }

    import_lines = [line for line in code.split('\n') if line.startswith('import') or line.startswith('from')]
    
    for line in import_lines:
        try:
            if line.startswith('import'):
                modules = [mod.strip() for mod in line.split('import')[1].split(',')]
            else:  # from ... import ...
                modules = [line.split()[1]]
            
            for module in modules:
                module = module.split(' as ')[0]
                
                if not any(module == lib or module.startswith(f"{lib}.") for lib in allowed_libraries):
                    print(f"Unauthorized dependency: {module}")
                    return False
                
                importlib.import_module(module)
        except ImportError as e:
            print(f"Missing dependency: {e.name}")
            return False
    return True

def generate_experiment_code(topic: str, model: ModelInterface) -> Dict[str, str]:
    prompt = f"""
    Given the research topic "{topic}", generate advanced Python code for an experiment and its corresponding plotting function.
    The experiment should be implementable and relevant to the topic. It must be robust and the quality of output should be similar to that of an experienced researcher.

    IMPORTANT: Use only the following libraries in your code:
        NumPy, SciPy, Pandas, Scikit-learn, Matplotlib, Seaborn, PyTorch, XGBoost, LightGBM, Statsmodels, Plotly, Pillow (PIL), OpenCV (cv2), NLTK, Gensim, Spacy, Joblib, Requests, deap, sympy, networkx, pytest, bokeh, dask, pyspark, skimage, imblearn !!

    Do not use any other libraries or dependencies, or something very bad will happen to innocent, baby kittens!!
    Ensure all necessary imports are included at the beginning of the code.

    Respond with a JSON object containing two keys:
    1. "experiment_code": The Python code for the main experiment
    2. "plot_code": The Python code for plotting the results
    """

    try:
        response = model.generate([{"role": "user", "content": prompt}])
        generated_code = json.loads(response)
        return ExperimentCodeOutput(**generated_code).dict()
    except Exception as e:
        logging.error(f"Error generating experiment code: {str(e)}")
        return {"experiment_code": "", "plot_code": ""}

def save_code_to_files(generated_code: Dict[str, str], base_dir: str) -> tuple:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    experiment_file = os.path.join(base_dir, 'experiment.py')
    plot_file = os.path.join(base_dir, 'plot.py')
    print("Experiment code generated successfully.", experiment_file)
    with open(experiment_file, 'w') as f:
        f.write(generated_code['experiment_code'])

    with open(plot_file, 'w') as f:
        f.write(generated_code['plot_code'])

    return experiment_file, plot_file

def execute_and_save_to_json(experiment_file: str, results_dir: str) -> dict:
    if not os.path.exists(experiment_file):
        raise FileNotFoundError(f"The specified file {experiment_file} does not exist.")

    result = subprocess.run(['python', experiment_file], capture_output=True, text=True)

    result_data = {
        "baseline_results": {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    }

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    json_path = os.path.join(results_dir, 'final_info.json')
    with open(json_path, 'w') as json_file:
        json.dump(result_data, json_file, indent=4)
    
    return result_data

def generate_code_fix(model: ModelInterface, error_message: str) -> Dict[str, str]:
    prompt = f"""
    The following error occurred while running an experiment script:

    {error_message}

    Please generate a new version of the experiment script to fix this error. Make sure the new code is functional and resolves the issue.

    Ensure the new experiment code is robust and relevant to the research topic.

    Respond with a JSON object containing the key "experiment_code" which should be the corrected Python code.
    """
    
    try:
        response = model.generate([{"role": "user", "content": prompt}])
        generated_code = json.loads(response)
        return ExperimentCodeOutput(**generated_code).dict()
    except Exception as e:
        logging.error(f"Error generating new experiment code: {str(e)}")
        return {"experiment_code": ""}

def generate_run_save_experiment(model: ModelInterface, base_dir: str):
    with open(os.path.join(base_dir, "topic.json"), "r") as f:
        topic = json.load(f)
    
    results_dir = os.path.join(base_dir, "run_0")
    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        generated_code = generate_experiment_code(topic, model)
        
        if check_dependencies(generated_code['experiment_code']):
            print("Experiment code generated successfully with all dependencies available.")
            experiment_file, plot_file = save_code_to_files(generated_code, base_dir)
            print(f"Experiment code saved to: {experiment_file}")
            
            try:
                result_data = execute_and_save_to_json(experiment_file, results_dir)
                stderr_output = result_data["baseline_results"]["stderr"]
                
                if stderr_output:
                    print("Error detected in experiment execution. Generating code fix...")
                    new_code = generate_code_fix(model, stderr_output)
                    
                    if new_code['experiment_code']:
                        new_experiment_file = os.path.join(base_dir, 'experiment_fixed.py')
                        with open(new_experiment_file, 'w') as f:
                            f.write(new_code['experiment_code'])
                        
                        print(f"New experiment code saved to: {new_experiment_file}")
                        result_data = execute_and_save_to_json(new_experiment_file, results_dir)
                        if result_data["baseline_results"]["stderr"]:
                            print(f"Failed to fix error after attempt {attempt + 1}.")
                        else:
                            print("Experiment executed successfully with the new code.")
                            return
                else:
                    print("Experiment executed successfully.")
                    return
            except Exception as e:
                print(f"Error executing experiment: {str(e)}")
                print("Retrying with new code generation...")
        else:
            print(f"Attempt {attempt + 1}: Generated code has unauthorized or missing dependencies. Retrying...")
        
        attempt += 1
    
    print(f"Failed to generate valid code after {max_attempts} attempts.")