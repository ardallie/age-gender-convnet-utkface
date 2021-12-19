"""
This module contains the main menu functionality.
"""
import sys

from convnet import console, convnet_utk


def create_data_split():
    """Option 1. Create data split."""
    console.print_menu()


def train_model():
    """Option 2. Train model."""
    console.print_menu()


def test_model():
    """Option 2. Train model."""
    console.print_menu()


def start():
    """
    Start the application and display the main menu.
    """
    __choose_menu = None
    console.print_menu()

    while True:

        __choose_menu = console.input_menu_choice()

        if __choose_menu == "":
            continue

        if __choose_menu == "1":
            convnet_utk.run_split_images()
            continue

        if __choose_menu == "2":
            convnet_utk.run_training()
            continue

        if __choose_menu == "3":
            convnet_utk.run_testing()
            continue

        if __choose_menu == "4":
            convnet_utk.run_gpu_info()
            continue

        if __choose_menu == "9":
            console.print_exit_app()
            sys.exit()

        else:
            console.print_menu()


if __name__ == "__main__":
    # main entry point - start the application
    start()
