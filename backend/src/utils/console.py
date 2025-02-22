CONSOLE_STYLE = {
    "ERROR": "\033[91m",
    "SECONDARY_ERROR": "\033[31m",
    "WARNING": "\033[93m",
    "SECONDARY_WARNING": "\033[33m",
    "SUCCESS": "\033[92m",
    "SECONDARY_SUCCESS": "\033[32m",
    "INFO": "\033[94m",
    "SECONDARY_INFO": "\033[96m",
    "END": "\033[0m",
    None: ""
}

class Style:
    """
    A class to represent a styled console message.
    """

    def __init__(self, style, message, auto_break=False, max_length=None):
        if style not in CONSOLE_STYLE:
            raise ValueError(f"Invalid style: {style}, must be one of:\n{CONSOLE_STYLE.keys()}")
        self.style = style

        if auto_break and max_length is not None:
            self.message = ""
            last_space = -1

            idx = 0
            for char in message:
                if idx % max_length == 0:
                    if last_space != -1:
                        cutted = self.message[idx:]
                        self.message = self.message[:idx]
                        self.message += "\n" + cutted.strip()
                        idx += 1 + len(cutted.strip()) - len(cutted)
                        char = char.strip()
                        last_space = -1
                    else:
                        self.message += "\n"
                        idx += 1
                self.message += char
                
                if char in [" ", "\n", ""]:
                    last_space = idx
                
                idx += len(char)
        
        elif auto_break:
            raise ValueError("If auto_break is True, max_length must be specified.")
        
        else:
            self.message = message

    def __repr__(self):
        if self.style is None:
            return self.message
        return f"{CONSOLE_STYLE[self.style]}{self.message}{CONSOLE_STYLE['END']}"
    
    def __str__(self):
        return self.__repr__()
    
    def __add__(self, other):
        return self.__repr__() + str(other)