from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterString,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFile,
                       QgsMessageLog,
                       Qgis)
import subprocess
import os
import shutil
import sys
import traceback

class ContinualLULCAlgorithm(QgsProcessingAlgorithm):
    PYTHON_PATH = 'PYTHON_PATH'
    MODEL_SCRIPT = 'MODEL_SCRIPT'
    DATA_DIR = 'DATA_DIR'
    YEAR = 'YEAR'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.PYTHON_PATH,
                'Python Interpreter Path (with PyTorch installed)',
                behavior=QgsProcessingParameterFile.File,
                fileFilter='Python (*.exe)',
                # Leave default blank to force user to supply correct interpreter
                defaultValue='' # Example: C:/Users/dhruv/anaconda3/envs/geo/python.exe
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFile(
                self.MODEL_SCRIPT,
                'Path to model/train.py',
                behavior=QgsProcessingParameterFile.File,
                fileFilter='Python Scripts (*.py)',
                # Auto-resolve relative to plugin location if left blank
                defaultValue=''
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFile(
                self.DATA_DIR,
                'Data Directory (containing Sentinel-2 images)',
                behavior=QgsProcessingParameterFile.Folder
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.YEAR,
                'Year to Process',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2024
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        python_path = self.parameterAsString(parameters, self.PYTHON_PATH, context)
        script_path = self.parameterAsString(parameters, self.MODEL_SCRIPT, context)
        data_dir = self.parameterAsString(parameters, self.DATA_DIR, context)
        year = self.parameterAsInt(parameters, self.YEAR, context)
        
        feedback.pushInfo(f"Starting Continual Learning process for year {year}...")

        # Determine python interpreter to use
        resolved_python = python_path
        if not resolved_python:
            # Try to auto-detect
            plugin_dir = os.path.dirname(__file__)
            project_root = os.path.normpath(os.path.join(plugin_dir, '..'))
            local_venv = os.path.join(project_root, 'lulc_env', 'Scripts', 'python.exe')
            if os.path.exists(local_venv):
                resolved_python = local_venv
                feedback.pushInfo(f"Auto-detected Python: {resolved_python}")
            else:
                feedback.reportError("No Python path provided and project venv not found.")
                return {}

        # Resolve script path if missing
        if not os.path.exists(script_path):
            plugin_dir = os.path.dirname(__file__)
            project_root = os.path.normpath(os.path.join(plugin_dir, '..'))
            alt = os.path.normpath(os.path.join(project_root, 'model', 'train.py'))
            if os.path.exists(alt):
                feedback.pushInfo(f"Auto-resolved MODEL_SCRIPT to: {alt}")
                script_path = alt
            else:
                feedback.reportError(f"Model script not found: {script_path}")
                return {}

        # Construct command and run
        cmd = [resolved_python, script_path, '--data_dir', data_dir, '--year', str(year)]
        feedback.pushInfo(f"Running command: {' '.join(cmd)}")

        try:
            # Create clean environment to avoid DLL conflicts with QGIS Python
            env = os.environ.copy()
            # Remove QGIS Python environment variables that might interfere
            for key in list(env.keys()):
                if 'PYTHONHOME' in key or 'PYTHONPATH' in key or 'QGIS' in key:
                    del env[key]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env
            )

            # Stream output to QGIS feedback
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    feedback.pushInfo(output.strip())

            rc = process.poll()
            if rc != 0:
                err = process.stderr.read()
                feedback.reportError(f"Training script error (exit code {rc}): {err}")
                return {}

        except Exception as e:
            tb = traceback.format_exc()
            feedback.reportError(f"Execution failed: {str(e)}\n{tb}")
            return {}

        feedback.pushInfo("Processing complete.")
        return {}

    def name(self):
        return 'run_continual_learning'

    def displayName(self):
        return 'Run Continual Learning Update'

    def group(self):
        return 'Analysis'

    def groupId(self):
        return 'analysis'

    def createInstance(self):
        return ContinualLULCAlgorithm()
