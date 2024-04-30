from customtkinter import *
from PIL import Image
from tkinter import messagebox
import json
import os



# Global variables for entry widgets
global username_entry
global password_entry

def store_credentials():
    username = username_entry.get()
    password = password_entry.get()
    
    credentials_list = []
    
    if os.path.exists("credentials.json"):
        with open("credentials.json", "r") as file:
            try:
                credentials_list = json.load(file)
            except json.JSONDecodeError:
                credentials_list = []
    
    credentials_exist = False
    for cred in credentials_list:
        if username == cred.get('username'):
            credentials_exist = True
            break
    
    if credentials_exist:
        messagebox.showinfo(title="Error", message="Username already exists. Please enter a new username.")
    else:
        credentials_list.append({"username": username, "password": password})
        with open("credentials.json", "w") as file:
            json.dump(credentials_list, file, indent=4)
        
        # Show success message
        response = messagebox.showinfo(title="Success", message="Username and password stored.")
        print (response)
        
        
        # Check response from messagebox if needed
        if response == "ok":
            app.destroy()
            from login import login
            
        
            # Close the current window or clear the fields
            username_entry.delete(0, END)
            password_entry.delete(0, END)
            
            # Navigate back to the login window
            # This assumes you have a function to destroy or hide the current window and show the login window.
            # For example, if you're using a function like `app.destroy()` to close the window,
            # make sure to call `Login_window()` afterwards to bring up the login page.
            # This part of your logic might need adjustment based on how your application's navigation is structured.
            login()
    
# This function is modified to destroy the current window and open a new one with a submit button.

def Signup_window():
    pass
app = CTk()
def Signups_window():
    global username_entry, password_entry
    
   
    app.geometry("730x470")
    app.resizable(0,0)
    
    # Load image data
    side_img_data = Image.open("Loginimage.png")
    email_icon_data = Image.open("email-icon.png")
    password_icon_data = Image.open("password-icon.png")
    
    # Create CTkImage instances for UI elements
    side_img = CTkImage(dark_image=side_img_data, light_image=side_img_data, size=(450, 480))
    email_icon = CTkImage(dark_image=email_icon_data, light_image=email_icon_data, size=(20,20))
    password_icon = CTkImage(dark_image=password_icon_data, light_image=password_icon_data, size=(17,17))
    
    # Place the side image in the main application window
    CTkLabel(master=app, text="", image=side_img).pack(expand=True, side="left")
    
    # Create a frame for login and registration elements
    frame = CTkFrame(master=app, width=320, height=480, fg_color="#ffffff")
    frame.pack_propagate(0)
    frame.pack(expand=True, side="right")
    
    # Add welcome and sign-in labels
    CTkLabel(master=frame, text="Welcome!", text_color="#601E88", anchor="w", justify="left", font=("Arial Bold", 24)).pack(anchor="w", pady=(50, 5), padx=(25, 0))
    CTkLabel(master=frame, text="Enter Your Details", text_color="#7E7E7E", anchor="w", justify="left", font=("Arial Bold", 12)).pack(anchor="w", padx=(25, 0))
    
    # Username and password fields with icons
    CTkLabel(master=frame, text="  Username:", text_color="#601E88", anchor="w", justify="left", font=("Arial Bold", 14), image=email_icon, compound="left").pack(anchor="w", pady=(38, 0), padx=(25, 0))
    username_entry = CTkEntry(master=frame, width=225, fg_color="#EEEEEE", border_color="#601E88", border_width=1, text_color="#000000")
    username_entry.pack(anchor="w", padx=(25, 0))
    
    CTkLabel(master=frame, text="  Password:", text_color="#601E88", anchor="w", justify="left", font=("Arial Bold", 14), image=password_icon, compound="left").pack(anchor="w", pady=(21, 0), padx=(25, 0))
    password_entry = CTkEntry(master=frame, width=225, fg_color="#EEEEEE", border_color="#601E88", border_width=1, text_color="#000000", show="*")
    password_entry.pack(anchor="w", padx=(25, 0))
    
    # Submit button
    CTkButton(master=frame, text="Submit", fg_color="#28A745", hover_color="#28A745", font=("Arial Bold", 12), text_color="#ffffff", width=225, command=store_credentials).pack(anchor="w", pady=(10, 0), padx=(25, 0))
    
    app.mainloop()
Signups_window()
