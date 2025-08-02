import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ShellAgent:
    """
    A more versatile ShellAgent that uses a single tool to execute commands
    after user approval, supporting different modes.
    """
    def __init__(self):
        # We now only have one function to call
        self.available_functions = {
            "execute_shell_command": self._execute_command_wrapper
        }
        # Pre-define a list of safe/allowed commands for 'safe' mode
        self.safe_commands = ["ls", "pwd", "df", "cat", "echo", "grep", "free"]
        # Pre-define a list of dangerous commands to block
        self.dangerous_commands = ["rm", "rmdir", "sudo", "reboot", "poweroff", "shutdown", "mv", "cp", "yay", "pacman"]

    def _execute_command_wrapper(self, command: str, args: list = None) -> str:
        """
        A wrapper that assembles and executes the command.
        This is the method the LLM's function_call will map to.
        """
        full_command = [command]
        if args:
            full_command.extend(args)
        
        return self._execute_command(full_command)

    def _execute_command(self, command: list[str]) -> str:
        """Helper to safely execute a shell command and return its output."""
        try:
            logging.info(f"Executing command: {' '.join(command)}")
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
        except FileNotFoundError:
            logging.error(f"Command '{command[0]}' not found.")
            return f"Error: Command '{command[0]}' not found. Is it installed and in your PATH?"
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"

    def get_tool_definitions(self) -> list[dict]:
        """
        Returns a single, universal tool definition for executing commands.
        """
        return [
            {
                "function_declarations": [
                    {
                        "name": "execute_shell_command",
                        "description": "Executes a shell command on the system. Use this tool for any task that requires interacting with the shell, such as listing files, checking disk space, or finding processes.",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "command": {
                                    "type": "STRING",
                                    "description": "The shell command to execute (e.g., 'ls', 'pwd', 'df')."
                                },
                                "args": {
                                    "type": "ARRAY",
                                    "description": "A list of arguments for the command (e.g., ['-lh', '/tmp'] for 'ls').",
                                    "items": {
                                        "type": "STRING"
                                    }
                                }
                            },
                            "required": ["command"]
                        }
                    }
                ]
            }
        ]

    def call_tool(self, tool_call: dict) -> str:
        """
        Dispatches the tool call to the corresponding Python function.
        In this case, it will always be the universal tool.
        """
        function_name = tool_call["name"]
        function_args = tool_call.get("args", {})

        if function_name in self.available_functions:
            func = self.available_functions[function_name]
            try:
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