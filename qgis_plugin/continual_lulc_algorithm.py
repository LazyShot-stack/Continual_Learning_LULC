from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterString,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFile,
                       QgsMessageLog,
                       Qgis)
import subprocess
import os

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
                defaultValue='C:/Users/dhruv/anaconda3/envs/geo/python.exe' # Example default
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFile(
                self.MODEL_SCRIPT,
                'Path to model/train.py',
                behavior=QgsProcessingParameterFile.File,
                fileFilter='Python Scripts (*.py)',
                defaultValue='d:/Sem_3/GIS_p/model/train.py'
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
        
        # Construct command
        # We need to modify train.py to accept arguments or pass them via env vars
        # For simplicity, let's assume we pass them as args
        cmd = [python_path, script_path, '--data_dir', data_dir, '--year', str(year)]
        
        feedback.pushInfo(f"Running command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
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
                feedback.reportError(f"Error: {err}")
                return {self.PYTHON_PATH: python_path} # Return something
                
        except Exception as e:
            feedback.reportError(f"Execution failed: {str(e)}")
            
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
