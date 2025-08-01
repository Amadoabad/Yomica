import subprocess
import json
import logging
import google.generativeai as genai 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ShellAgent:
    """
    Interprets natural language queries into safe, executable shell commands.
    Uses a predefined set of tools/functions for safety.
    """

    def __init__(self):
        self.available_functions = {
            "get_disk_space": self.get_disk_space,
            "list_running_processes": self.list_running_processes,
            "get_current_directory": self.get_current_directory,
            "list_files_in_directory": self.list_files_in_directory,
        }

    def _execute_command(self, command: list[str]) -> str:
        """Helper to safely execute a shell command and return its output."""
        try:
            logging.info(f"Executing command: {' '. join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            return result.stdout.strip()
        
        except subprocess.CalledProcessError as e:
            logging.error(f"Command '{' '.join(command)}' failed with error: {e.stderr.strip()}")
            return f"Error executing command: {e.stderr.strip()}"
        
        except subprocess.TimeoutExpired:
            logging.error(f"Command '{' '.join(command)}' timed out.")
            return f"Error: Command timed out after 10 seconds."
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"
        
    def get_disk_space(self) -> str:
        """Returns informaiton about disk space usage."""
        return self._execute_command(["df", "-h"])
    
    def list_running_processes(self, top_n: int = 10, sort_by: str = "memory"):
        """
        Lists running processes, optionally sorted by memory or CPU, and limited to top N.
        Args:
            top_n: Number of processes to list.
            sort_by: 'memory' or 'cpu'.
        """

        if sort_by not in ["memory", "cpu"]:
            return "Error: Invalid sort_by option. Must be 'memory' or 'cpu'."
        if not isinstance(top_n, int) or top_n <=0:
            if isinstance(top_n, float) and top_n >0:
                top_n = int(top_n)
            else:
                return f"Error: top_n must be a positive integer. {top_n}"
        
        sort_param = r"-%mem" if sort_by == "memory" else r"-%cpu"
        command = ['ps', 'aux', f'--sort={sort_param}']
        output = self._execute_command(command)
        lines = output.splitlines()
        if len(lines) > top_n +1: # +1 for header
            return "\n".join(lines[:top_n+1])
        return output
    
    def get_current_directory(self) -> str:
        """Returnss the current working directory."""
        return self._execute_command(["pwd"])
    
    def list_files_in_directory(self, path: str = ".",show_hidden: bool = True, long_format: bool = False) -> str:
        """
        Lists files and directories in a given path.
        Args:
            path: The directory path (default: current directory).
            long_format: If True, uses 'ls -l'.
        """

        command = ['ls']
        if long_format:
            command.append('-l')
        
        if show_hidden:
            command.append('-A')
        
        command.append(path)

        return self._execute_command(command)
    
    def get_tool_definitions(self) -> list[dict]:
        """
        Returns the tool definitions in a format suitable for the Gemini API.
        This describes the functions the LLM can call.
        """
        return [
            {
                "function_declarations": [ # Functions are wrapped in a list under "function_declarations"
                    {
                        "name": "get_disk_space",
                        "description": "Get information about disk space usage on the system.",
                        "parameters": {
                            "type": "OBJECT", # Use uppercase for type string
                            "properties": {},
                            "required": []
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "list_running_processes",
                        "description": "List running processes, sortable by memory or CPU usage, showing top N processes.",
                        "parameters": {
                            "type": "OBJECT", # Use uppercase for type string
                            "properties": {
                                "top_n": {
                                    "type": "INTEGER", # Use uppercase for type string
                                    "description": "The number of top processes to list, must be positive INT not a float(10 for top 10)."
                                },
                                "sort_by": {
                                    "type": "STRING", # Use uppercase for type string
                                    "description": "The criteria to sort processes by ('memory' or 'cpu').",
                                    "enum": ["memory", "cpu"]
                                }
                            },
                            "required": []
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "get_current_directory",
                        "description": "Get the current working directory.",
                        "parameters": {
                            "type": "OBJECT", # Use uppercase for type string
                            "properties": {},
                            "required": []
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "list_files_in_directory",
                        "description": "List files and directories in a specified path, with an option for long format.",
                        "parameters": {
                            "type": "OBJECT", # Use uppercase for type string
                            "properties": {
                                "path": {
                                    "type": "STRING", # Use uppercase for type string
                                    "description": "The directory path to list files from. Defaults to current directory if not provided."
                                },
                                "long_format": {
                                    "type": "BOOLEAN", # Use uppercase for type string
                                    "description": "If true, lists in long format (ls -l)."
                                }
                            },
                            "required": []
                        }
                    }
                ]
            }
        ]

    def call_tool(self, tool_call:dict) -> str:
        """
        Dispatches the tool call to the corresponding Python function.
        Args:
            tool_call: A dictionary representing the tool call from the LLM,
                       e.g., {"name": "get_disk_space", "args": {}}.
        Returns:
            The result of the tool execution as a string.
        """
        function_name = tool_call["name"]
        function_args = tool_call.get("args", {})

        if function_name in self.available_functions:
            func = self.available_functions[function_name]
            try:
                # Call the function with unpacked arguments
                result = func(**function_args)
                return result
            except TypeError as e:
                logging.error(f"Error calling function {function_name} with args {function_args}: {e}")
                return f"Error: Invalid arguments for command '{function_name}'. Details: {e}"
            
            except Exception as e:
                logging.error(f"An unexpected error occurred during tool execution: {e}")
                return f"An unexpected error occurred during tool execution: {e}"
            
        else:
            return f"Error: Function '{function_name}' not found or not supported."

