import time
from os_computer_use.streaming import Sandbox, DisplayClient
from os_computer_use.browser import Browser
from os_computer_use.sandbox_agent import SandboxAgent
from os_computer_use.logging import Logger
import asyncio
import argparse

import os
from dotenv import load_dotenv

logger = Logger()

# Load environment variables from .env file
load_dotenv()

# Configure E2B
os.environ["E2B_API_KEY"] = os.getenv("E2B_API_KEY")


async def start(user_input=None, output_dir=None):
    sandbox = None
    client = None
    
    try:
        sandbox = Sandbox()

       # The display server won't work on desktop-dev-v2 since ffmpeg is not installed
    #     client = DisplayClient(output_dir)
    #     print("Starting the display server...")
    #     stream_url = sandbox.start_stream()
    #     print("(The display client will start in five seconds.)")
    #    # If the display client is opened before the stream is ready, it will close immediately
    #     await client.start(stream_url, user_input or "Sandbox", delay=5)
        agent = SandboxAgent(sandbox, output_dir)
        print("Starting the VNC server...")
        sandbox.stream.start()
        vnc_url = sandbox.stream.get_url()
        #print(sandbox.sandbox_id)
        #sandbox.co
        print("Starting the VNC client...")
        browser = Browser()
        if(not browser.is_running):
            browser.open(vnc_url)
        while True:
            # Ask for user input, and exit if the user presses ctl-c
            #print(sandbox.)
            print(sandbox.stream)
            if user_input is None:
                try:
                    user_input = input("USER: ")
                except KeyboardInterrupt:
                    break
            # Run the agent, and go back to the prompt if the user presses ctl-c
            else:
                try:
                    agent.run(str(user_input))
                    user_input = None
                except KeyboardInterrupt:
                    user_input = None
                except Exception as e:
                    logger.print_colored(f"An error occurred: {e}", "red")
                    user_input = None
            sandbox.set_timeout(600, 10)
            logger.log(sandbox.sandbox_id, "gray")
            logger.log(sandbox.get_host(8080), "gray")
            logger.log(sandbox.get_info(), "gray")
        print(agent.messages)

    finally:
        #if client:
        #    print("Stopping the display client...")
        #    try:
        #        await client.stop()
        #    except Exception as e:
        #        print(f"Error stopping display client: {str(e)}")

        if sandbox:
            print("Stopping the sandbox...")
            try:
                sandbox.kill()
            except Exception as e:
                print(f"Error stopping sandbox: {str(e)}")

        #if client:
        #    print("Saving the stream as mp4...")
        #    try:
        #        await client.save_stream()
        #    except Exception as e:
        #        print(f"Error saving stream: {str(e)}")

        print("Stopping the VNC client...")
        try:
            browser.close()
        except Exception as e:
            print(f"Error stopping VNC client: {str(e)}")


def initialize_output_directory(directory_format):
    run_id = 1
    while os.path.exists(directory_format(run_id)):
        run_id += 1
    os.makedirs(directory_format(run_id), exist_ok=True)
    return directory_format(run_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="User prompt for the agent")
    args = parser.parse_args()
    print(f"{args.prompt}")
    output_dir = initialize_output_directory(lambda id: f"./output/run_{id}")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start(output_dir=output_dir, user_input=args.prompt))

