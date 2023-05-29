import importlib

class installer():
    def __init__(self):
        packages = [("numpy", "numpy"), ("time", "time"), ("gzip", "gzip"), ("customtkinter", "customtkinter"), ("PIL", "pillow"), ("pickle", "pickle"), ("matplotlib", "matplotlib"), ("os", "os")]
        for i in packages:
            self.install(i)
    
    def install(self, package):
        try:
            importlib.import_module(package[0])
        except ImportError:
            import subprocess
            print(f"Installing {package[0]}...")
            subprocess.check_call(["pip", "install", package[1]])
