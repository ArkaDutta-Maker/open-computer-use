import time
from os_computer_use.config import vision_model, action_model, grounding_model
from os_computer_use.llm_provider import Message
from os_computer_use.logging import logger
from os_computer_use.grounding import draw_big_dot

import shlex
import os
import tempfile
from PIL import Image
import json

TYPING_DELAY_MS = 60
TYPING_GROUP_SIZE = 50

tools = dict()
    # "stop": {
    #     "description": "Indicate that the task has been completed.",
    #     "params": {},
    # }    



class SandboxAgent:

    def __init__(self, sandbox, output_dir=".", save_logs=True):
        super().__init__()
        self.messages = []  # Agent memory
        self.sandbox = sandbox  # E2B sandbox
        self.latest_screenshot = None  # Most recent PNG of the scren
        self.image_counter = 0  # Current screenshot number
        self.tmp_dir = tempfile.mkdtemp()  # Folder to store screenshots
        # Set the log file location
        if save_logs:
            logger.log_file = f"{output_dir}/log.html"

        print("The agent will use the following actions:")
        for action, details in tools.items():
            param_str = ", ".join(details.get("params").keys())
            print(f"- {action}({param_str})")

    def call_function(self, name, arguments):

        func_impl = getattr(self, name.lower()) if name.lower() in tools else None
        if func_impl:
            try:
                result = func_impl(**arguments) if arguments else func_impl()
                return result
            except Exception as e:
                return f"Error executing function: {str(e)}"
        else:
            return "Function not implemented."

    def tool(description, params):
        def decorator(func):
            tools[func.__name__] = {"description": description, "params": params}
            return func

        return decorator

    def save_image(self, image, prefix="image"):
        self.image_counter += 1
        filename = f"{prefix}_{self.image_counter}.png"
        filepath = os.path.join(self.tmp_dir, filename)
        if isinstance(image, Image.Image):
            image.save(filepath)
        else:
            with open(filepath, "wb") as f:
                f.write(image)
        return filepath

    def screenshot(self):
        file = self.sandbox.screenshot()
        filename = self.save_image(file, "screenshot")
        logger.log(f"screenshot {filename}", "gray")
        self.latest_screenshot = filename
        with open(filename, "rb") as image_file:
            return image_file.read()

    @tool(
        description="Run a shell command and return the result.",
        params={"command": "Shell command to run synchronously"},
    )
    def run_command(self, command):
        result = self.sandbox.commands.run(command, timeout=60)
        stdout, stderr = result.stdout, result.stderr
        if stdout and stderr:
            return stdout + "\n" + stderr
        elif stdout or stderr:
            return stdout + stderr
        else:
            return "The command finished running."

    @tool(
        description="Run a shell command in the background.",
        params={"command": "Shell command to run asynchronously"},
    )
    def run_background_command(self, command):
        self.sandbox.commands.run(command, background=True)
        return "The command has been started."

    @tool(
        description="Send a key or combination of keys to the system.",
        params={"name": "Key or combination (e.g. 'Return', 'Ctl-C')"},
    )
    def send_key(self, name):
        self.sandbox.press(name)
        return "The key has been pressed."

    @tool(
        description="Type a specified text into the system.",
        params={"text": "Text to type"},
    )
    def type_text(self, text):
        self.sandbox.write(text, chunk_size=TYPING_GROUP_SIZE, delay_in_ms=TYPING_DELAY_MS)
        return "The text has been typed."

    def click_element(self, query, click_command, action_name="click"):
        """Base method for all click operations"""
        self.screenshot()
        position = grounding_model.call(query, self.latest_screenshot)
        dot_image = draw_big_dot(Image.open(self.latest_screenshot), position)
        filepath = self.save_image(dot_image, "location")
        logger.log(f"{action_name} {filepath})", "gray")

        x, y = position
        self.sandbox.move_mouse(x, y)
        click_command()
        return f"The mouse has {action_name}ed."

    @tool(
        description="Click on a specified UI element.",
        params={"query": "Item or UI element on the screen to click"},
    )
    def click(self, query):
        return self.click_element(query, self.sandbox.left_click)

    @tool(
        description="Double click on a specified UI element.",
        params={"query": "Item or UI element on the screen to double click"},
    )
    def double_click(self, query):
        return self.click_element(query, self.sandbox.double_click, "double click")

    @tool(
        description="Right click on a specified UI element.",
        params={"query": "Item or UI element on the screen to right click"},
    )
    def right_click(self, query):
        return self.click_element(query, self.sandbox.right_click, "right click")

    def append_screenshot(self):
        return vision_model.call(
            [
                *self.messages,
                Message(
                    [
                         self.screenshot(),                        
                         #"This image shows the current display of the computer. Please respond in the following format:\n"
                        # "The objective is: [put the objective here]\n"
                        # "On the screen, I see: [an extensive list of everything that might be relevant to the objective including windows, icons, menus, apps, and UI elements]\n"
                        # "This means the objective is: [complete|not complete]\n\n"
                        # "(Only continue if the objective is not complete.)\n"
                        # "The next step is to [click|type|run the shell command] [put the next single step here] in order to [put what you expect to happen here].",

                        """ 
                        This image shows the current display of the computer.
                        Analyze thoroughly and donot give any unwanted information.
                        The computer is running a Linux operating system.
                        You are a vision assistant whose job is to inspect a screenshot of my desktop and help me complete a goal.
                            
                            Format your answer **exactly** as follows:
                            The objective is: [put the objective here]

                            On the screen, I see:
                            - [a detailed, bullet-point list of every relevant window, icon, menu, app and UI element that relates to the objective]

                            This means the objective is: [complete | not complete]

                            (Only continue if the objective is not complete.)

                            The next step is to [click | type | run the shell command] [the one single action to take] in order to [what you expect will happen next]."""
                            ,
                    ],
                    role="user",
                ),
            ]
        )

    def run(self, instruction):

        self.messages.append(Message(f"OBJECTIVE: {instruction}"))
        logger.log(f"USER: {instruction}", print=False)

        should_continue = True
        cnt = 0
        while should_continue:
            self.sandbox.set_timeout(600, 600)
            v = self.sandbox.list()
            print(v)
            content, tool_calls = action_model.call(
                [
                    Message(
                        """
                        You are a helpful desktop assistant.
                        When assisting the user:
                            1. Determine if the user needs to open a local application or visit a website.
                            2. If it's a website, invoke the browser tool(**globe icon**) with an HTTPS URL. 
                            3. Always decide whether a tool call is required.
                            4. If you call a tool, output a JSON payload that exactly matches its signature.
                            5. After the tool returns its result, resume your reasoning and reply in natural language.
                            6. If no tool is needed, answer directly.""",
                        role="system",
                    ),
                    *self.messages,
                    Message(
                        logger.log(f"THOUGHT: {self.append_screenshot()}", "green")
                    ),
                    Message(
                        "I will now use tool calls to take these actions.",
                    ),
                ],
                tools,
            )

            if content:
                self.messages.append(Message(logger.log(f"THOUGHT: {content}", "blue")))

            should_continue = True
            logger.log(f"TOOL CALLS: {tool_calls}", "red")
            if(tool_calls is None):
                cnt += 1
            
            if cnt > 3:
                should_continue = False
            
            for tool_call in tool_calls:
                name, parameters = tool_call.get("name"), tool_call.get("parameters")
                should_continue = name != "stop"
                logger.log(f"stop: {not should_continue}", color="red")
                if not should_continue:
                    break
                # Print the tool-call in an easily readable format
                logger.log(f"ACTION: {name} {str(parameters)}", "red")
                # Write the tool-call to the message history using the same format used by the model
                self.messages.append(Message(json.dumps(tool_call)))
                result = self.call_function(name, parameters)

                self.messages.append(
                    Message(logger.log(f"OBSERVATION: {result}", "yellow"))
                )
            #logger.log(self.messages[-1].get("content").lower(), color="blue")
