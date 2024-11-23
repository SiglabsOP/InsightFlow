import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
import re
import os
import psutil
import subprocess
import sys


def analyze_and_update_ticker():
    file_path = "charter.py"  # Adjust if located elsewhere (c) SIG LABS 2024 target 7.0

    try:
        # Step 1: Read the file content
        with open(file_path, 'r') as file:
            content = file.read()

        # Step 2: Locate the current ticker value
        ticker_match = re.search(r"ticker\s*=\s*['\"]([A-Z]+)['\"]", content, re.IGNORECASE)
        if not ticker_match:
            messagebox.showerror("Error", "Ticker value not found in the script!")
            sys.exit()  # Exit if the ticker is not found

        current_ticker = ticker_match.group(1)

        # Step 3: Count occurrences of the ticker
        ticker_count = content.count(current_ticker)

        # Step 4: Display results in a GUI
        def show_analysis():
            root = tk.Tk()
            root.title("Asset Analysis")

            # Create text area for results
            analysis_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, font=("Helvetica", 12))
            analysis_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
            analysis_text.insert(
                tk.END,
                f"Please enter the new ticker so I can create charts for you. Current Chart: {current_ticker}\n",
            )
            analysis_text.configure(state="disabled")

            # Prompt for new ticker
            def replace_ticker():
                new_ticker = simpledialog.askstring(
                    "What asset would you like to analyze? Please Enter a valid Ticker",
                    "Enter your asset:",
                    parent=root,
                )
                if new_ticker:
                    # Normalize ticker to uppercase
                    new_ticker = new_ticker.upper()

                    nonlocal content
                    replacement_count = content.count(current_ticker)
                    content = content.replace(current_ticker, new_ticker)

                    # Save the modified file
                    with open(file_path, 'w') as updated_file:
                        updated_file.write(content)

                    # Check and restart `charter.py`
                    restart_charter()

                    # Exit after work is done
                    sys.exit()

            # Add buttons
            replace_button = tk.Button(root, text="Analyze Asset", command=replace_ticker, bg="green", fg="white")
            replace_button.pack(pady=10)

            quit_button = tk.Button(root, text="Close", command=lambda: [root.destroy(), sys.exit()], bg="red", fg="white")
            quit_button.pack(pady=5)

            root.mainloop()

        show_analysis()

    except FileNotFoundError:
        messagebox.showerror("Error", f"File '{file_path}' not found!")
        sys.exit()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        sys.exit()



 

def restart_charter():
    """Check if 'charter.py' is running. If so, kill it and restart it completely detached."""
    script_name = "charter.py"
    python_executable = sys.executable  # Current Python interpreter
    script_path = os.path.abspath(script_name)  # Full path to charter.py
    script_dir = os.path.dirname(script_path)  # Directory containing charter.py
    process_found = False

    # Check for running processes
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = process.info.get('cmdline')
            if cmdline and len(cmdline) > 1 and script_name in cmdline[-1]:
                # Kill the process
                print(f"Found running instance of {script_name}. Killing it...")
                process.terminate()
                process.wait(timeout=5)
                process_found = True
                print(f"Successfully terminated {script_name}.")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except psutil.TimeoutExpired:
            print(f"Process {process.pid} did not terminate in time. Force killing.")
            process.kill()

    # Start the script in a detached manner
    if process_found:
        print(f"Restarting {script_name}...")
    else:
        print(f"{script_name} is not running. Starting it now...")

    try:
        # Define subprocess options
        kwargs = {
            "cwd": script_dir,  # Set working directory to script folder
             # "stdout": open(os.path.join(script_dir, "charter_stdout.log"), "w"),
             # "stderr": open(os.path.join(script_dir, "charter_stderr.log"), "w"),
            "close_fds": True  # Close file descriptors
        }

        if os.name == 'nt':  # Windows
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:  # Unix-like systems
            kwargs["preexec_fn"] = os.setsid

        # Launch the script
        subprocess.Popen([python_executable, script_path], **kwargs)
        print(f"{script_name} started successfully in {script_dir}.")

    except Exception as e:
        print(f"Failed to start {script_name}: {e}")





if __name__ == "__main__":
    analyze_and_update_ticker()
