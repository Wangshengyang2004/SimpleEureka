class RUNNING_TOO_LONG_ERROR(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"Code run is taking too long to execute. {self.message}"

class CODE_EXECUTION_SYNTAXERROR(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"Code execution error. Check syntax setting {self.message}"
    
class CODE_EXECUTION_VALUEERROR(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"Code execution error. Check grammar and varible setting {self.message}"