import tkinter as tk

def open(user):
    
    # Create a Tkinter window
    window = tk.Tk()

    # Set the window title
    window.title("Tkinter Window")

    # Set the window size (width x height)
    window.geometry("400x300")

    # Create a label widget
    label = tk.Label(window, text=f"Hello, {user}!")
    label.pack(pady=20)  # Add some padding around the label

    # Run the Tkinter event loop
    return window