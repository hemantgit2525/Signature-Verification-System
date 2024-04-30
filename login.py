from customtkinter import *
from PIL import Image
from tkinter import messagebox
import json

# This function is modified to destroy the current window and open a new one with a submit button.
def login_window():
    global user_entry
    global password_entry
    user = user_entry.get()
    password = password_entry.get()
    
    
    
    # Read credentials from the JSON file
    with open('credentials.json') as f:
        credentials = json.load(f)
        
    # Both Username and Password is empty
    if not user or not password:
        print("Username and Password is empty\nPlease enter Username and Password")
        messagebox.showinfo(title="Error", message="Username and Password is empty\nPlease enter Username and Password.")
    
    # Check if the entered username and password match any entry in the credentials list
    for cred in credentials:
        if user == cred['username'] and password == cred['password']:
            app.destroy()  # Destroy the current main window
            from signworking import image_uploader
            image_uploader()
            return
              # Exit the function if credentials match
    
    print('Error: Username or password is incorrect.')
    
    
def Signup_window():
    global user
    global password
    app.destroy()  # Destroy the current main window
    from signup import Signups_window
    Signups_window()
    ''' new_app = CTk()  # Create a new main window instance
    new_app.geometry("730x470")
    new_app.title("New Window")

    # Adding a submit button to the new window
    CTkButton(new_app, text="Submit", fg_color="#007BFF", hover_color="#007BFF",
              font=("Arial Bold", 12), text_color="#ffffff").pack(pady=220) '''

        
    #new_app.mainloop()  # Start the main loop for the new window

# Initialize the main application window
app = CTk()
user_entry = ""
password_entry = ""
def login():
    global user_entry
    global password_entry
   
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
    CTkLabel(master=frame, text="Sign in to your account", text_color="#7E7E7E", anchor="w", justify="left", font=("Arial Bold", 12)).pack(anchor="w", padx=(25, 0))

    # Username and password fields with icons
    CTkLabel(master=frame, text="  Username:", text_color="#601E88", anchor="w", justify="left", font=("Arial Bold", 14), image=email_icon, compound="left").pack(anchor="w", pady=(38, 0), padx=(25, 0))
    user_entry = CTkEntry(master=frame, width=225, fg_color="#EEEEEE", border_color="#601E88", border_width=1, text_color="#000000")
    user_entry.pack(anchor="w", padx=(25, 0))

    CTkLabel(master=frame, text="Password:", text_color="#601E88", anchor="w", justify="left", font=("Arial Bold", 14), image=password_icon, compound="left").pack(anchor="w", pady=(21, 0), padx=(25, 0))
    password_entry = CTkEntry(master=frame, width=225, fg_color="#EEEEEE", border_color="#601E88", border_width=1, text_color="#000000", show="*")
    password_entry.pack(anchor="w", padx=(25, 0))

    # Login and registration buttons
    CTkButton(master=frame, text="Login", fg_color="#007BFF", hover_color="#007BFF", font=("Arial Bold", 12), text_color="#ffffff", width=225, command=login_window).pack(anchor="w", pady=(40, 0), padx=(25, 0))
    CTkButton(master=frame, text="Signup", fg_color="#28A745", hover_color="#28A745", font=("Arial Bold", 12), text_color="#ffffff", width=225, command=Signup_window).pack(anchor="w", pady=(10, 0), padx=(25, 0))
    app.attributes("-topmost",True)
    app.mainloop()
    # return app
    
# login()
