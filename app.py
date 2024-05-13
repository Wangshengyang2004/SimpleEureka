"""
Features will be added:
1. Manage files and folders, meaning you can delete, download zipeed files
2. Watch Training Videos
3. Parse Tensorboard logs
"""
import os
import streamlit as st
from io import BytesIO

def app():
    st.title('Interactive File and Folder Viewer')
    
    base_dir = './results'
    
    # This will hold the current path. Initialize with base directory.
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = base_dir

    # This will track the file currently being displayed
    if 'open_file' not in st.session_state:
        st.session_state['open_file'] = None

    # Function to navigate to a directory
    def navigate_to(directory):
        st.session_state['current_path'] = directory
        st.session_state['open_file'] = None  # Close any open file when navigating

    # Function to toggle file display
    def toggle_file_display(file_path):
        if st.session_state['open_file'] == file_path:
            st.session_state['open_file'] = None
        else:
            st.session_state['open_file'] = file_path

    # Function to display contents of a file
    def display_file_content(file_path):
        _, ext = os.path.splitext(file_path)
        if ext in ['.txt', '.py', '.log']:
            with open(file_path, "r") as file:
                content = file.read()
                st.code(content, language=ext[1:])
        else:
            st.error("Unsupported file format.")

    # Sidebar with navigation links
    st.sidebar.title("Navigation")
    # Go back to base directory
    if st.sidebar.button('Go to Base Directory'):
        navigate_to(base_dir)

    # Navigational part: Construct the folder path navigation
    parts = os.path.relpath(st.session_state['current_path'], base_dir).split(os.sep)
    path_accum = base_dir
    for i, part in enumerate(parts):
        if st.sidebar.button(f"{'→' * i} {part}"):
            path_accum = os.path.join(base_dir, *parts[:i + 1])
            navigate_to(path_accum)
            break

    # List current directory
    current_files = []
    current_dirs = []
    for item in os.listdir(st.session_state['current_path']):
        item_path = os.path.join(st.session_state['current_path'], item)
        if os.path.isdir(item_path):
            current_dirs.append(item)
        else:
            current_files.append(item)
    
    # Display directories and files in the main area
    for directory in sorted(current_dirs):
        if st.button(f"📁 {directory}", key=directory):
            navigate_to(os.path.join(st.session_state['current_path'], directory))
    
    for file in sorted(current_files):
        if st.button(f"📄 {file}", key=file):
            toggle_file_display(os.path.join(st.session_state['current_path'], file))

    # Display file content if a file is open
    if st.session_state['open_file']:
        display_file_content(st.session_state['open_file'])

if __name__ == "__main__":
    st.set_page_config(page_title="File Explorer", layout="wide")
    app()
