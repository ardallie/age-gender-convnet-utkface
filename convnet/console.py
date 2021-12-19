"""
Contains consolidated console messaging (print and input).
"""


def print_menu():
    """Displays the application's Main Menu"""
    print()
    print(".........................................")
    print(".               Main Menu:              .")
    print(".........................................")
    print("\t 1  Split and copy images")
    print("\t 2  Train the model")
    print("\t 3  Test the model")
    print("\t 4  Display GPU and CUDA info")
    print("\t 9  Exit")
    print("\t 0  Display main menu")
    print()


def print_exit_app():
    """Display a message on exit"""
    print("Exiting now. Thanks.")


def input_menu_choice():
    """get the menu option"""
    return input("[MENU]  Enter your choice (0 to show menu): ").strip()
