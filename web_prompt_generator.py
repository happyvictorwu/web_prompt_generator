import streamlit as st
import os
import json
import chardet
import zipfile
import tempfile
import io
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, IO, Set
import logging # Added for better error reporting

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
PROJECT_DATA_FILE = "streamlit_project_config_zip.json"
DEFAULT_EXTENSIONS = [".md", ".txt"]
DEFAULT_OUTPUT_PATTERN = "{project_name}_Prompt.txt"
CHUNK_SIZE_ENCODING = 1024 * 10  # Read 10KB for encoding detection

# --- Core Processing Logic ---

def detect_encoding(file_path: str) -> str:
    """Detects file encoding using chardet, falling back to UTF-8."""
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(CHUNK_SIZE_ENCODING)
            if not raw_data: return 'utf-8'
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result and result['encoding'] and result['confidence'] > 0.7 else 'utf-8'
            # Handle potentially problematic encodings detected by chardet
            if encoding.lower() in ['ascii']:
                encoding = 'utf-8' # Treat ascii as utf-8 for broader compatibility
            elif encoding.lower() in ['gb2312', 'gbk', 'big5']:
                 # Allow common CJK encodings if detected with high confidence, otherwise fallback
                 if result['confidence'] < 0.9:
                     logger.warning(f"Low confidence ({result['confidence']:.2f}) for {encoding} in '{os.path.basename(file_path)}'. Falling back to utf-8.")
                     encoding = 'utf-8'
            elif encoding is None:
                 logger.warning(f"Encoding detection returned None for '{os.path.basename(file_path)}'. Falling back to utf-8.")
                 encoding = 'utf-8'

            return encoding
    except (IOError, OSError) as e:
        logger.warning(f"Encoding detection failed for '{os.path.basename(file_path)}': {e}. Falling back to utf-8.")
        st.warning(f"Encoding detection failed for '{os.path.basename(file_path)}': {e}. Falling back to utf-8.")
        return 'utf-8'
    except Exception as e:
        logger.error(f"Unexpected error during encoding detection for '{os.path.basename(file_path)}': {e}", exc_info=True)
        st.warning(f"Unexpected error during encoding detection for '{os.path.basename(file_path)}'. Falling back to utf-8.")
        return 'utf-8'


def _read_file_content(full_path: str, encoding: str) -> str:
    """Reads file content, compresses newlines, and handles errors."""
    try:
        with open(full_path, 'r', encoding=encoding, errors='replace') as file:
            # Read line by line to potentially handle very large files better
            content_buffer = io.StringIO()
            for line in file:
                # Replace newlines within the line itself, then add a single space
                processed_line = line.replace('\n', ' ').replace('\r', ' ')
                content_buffer.write(processed_line)
            content = content_buffer.getvalue()
            return content
    except (IOError, OSError, UnicodeDecodeError) as e:
        logger.warning(f"Error reading file '{os.path.basename(full_path)}' with encoding '{encoding}': {e}")
        st.warning(f"Error reading file '{os.path.basename(full_path)}': {e}")
        return f"Error reading file: {e}"
    except Exception as e:
        logger.error(f"Unexpected error reading file '{os.path.basename(full_path)}': {e}", exc_info=True)
        st.warning(f"Unexpected error reading file '{os.path.basename(full_path)}'.")
        return f"Unexpected error reading file: {e}"

def get_structure_and_content_from_dir(
    base_directory: str,
    allowed_extensions: Tuple[str, ...]
) -> Tuple[str, str]:
    """
    Walks a directory, extracts structure, reads content of allowed files,
    and returns structure and content strings. Handles multiple top-level items.
    """
    directory_structure = io.StringIO()
    file_contents = io.StringIO()
    processed_files_count = 0
    encountered_extensions: Set[str] = set() # Track extensions encountered

    try:
        # List top-level items to determine the display root
        top_level_items = os.listdir(base_directory)
        display_root_name = "Extracted Content Root" # Default if multiple items or just files at root

        # Determine a sensible relative path root. If there's exactly one directory
        # at the top level, use that. Otherwise, use the base_directory.
        content_root_for_relpath = base_directory
        single_top_level_dir = None
        top_level_dirs = [item for item in top_level_items if os.path.isdir(os.path.join(base_directory, item))]

        if len(top_level_dirs) == 1 and len(top_level_items) == 1:
             single_top_level_dir = os.path.join(base_directory, top_level_dirs[0])
             content_root_for_relpath = single_top_level_dir
             display_root_name = top_level_dirs[0]
             logger.info(f"Using single top-level directory '{display_root_name}' as content root.")

        directory_structure.write(f"{display_root_name}/\n")

        # Walk from the base directory always
        for root, dirs, files in os.walk(base_directory, topdown=True):
            # Filter and sort directories and files - ignore common metadata/hidden
            dirs[:] = sorted([d for d in dirs if not d.startswith('.') and not d.lower() == '__macosx' and not d.startswith('__MACOSX')])
            files = sorted([f for f in files if not f.startswith('.') and f != '.DS_Store'])

            # Calculate relative path from the determined content root
            try:
                 # Use base_directory for walking, but content_root_for_relpath for display paths
                relative_root_dir_display = os.path.relpath(root, content_root_for_relpath)
            except ValueError:
                 # This might happen if root is outside content_root_for_relpath, shouldn't occur with this logic
                 relative_root_dir_display = os.path.relpath(root, base_directory)
                 logger.warning(f"Relpath calculation resulted in unexpected path: {relative_root_dir_display}")

            # Calculate depth based on the display relative path
            depth = relative_root_dir_display.count(os.sep) if relative_root_dir_display != '.' else -1

            # Add directory entry to structure (if not the root itself)
            if relative_root_dir_display != ".":
                indent = '    ' * (depth + 1) # +1 because root is level 0
                directory_structure.write(f"{indent}{os.path.basename(root)}/\n")

            # Process files
            file_indent_level = depth + (1 if relative_root_dir_display == "." else 2)
            file_indent = '    ' * file_indent_level
            for file_name in files:
                directory_structure.write(f"{file_indent}{file_name}\n")

                _, file_ext = os.path.splitext(file_name)
                file_ext_lower = file_ext.lower()
                encountered_extensions.add(file_ext_lower) # Track all extensions

                if file_ext_lower in allowed_extensions:
                    full_file_path = os.path.join(root, file_name)
                    try:
                        relative_file_path_display = os.path.relpath(full_file_path, content_root_for_relpath)
                    except ValueError:
                        relative_file_path_display = os.path.relpath(full_file_path, base_directory)


                    encoding = detect_encoding(full_file_path)
                    logger.info(f"Reading '{relative_file_path_display}' with encoding '{encoding}'")
                    file_content = _read_file_content(full_file_path, encoding)

                    file_contents.write(f"\n\n> Source Path: {relative_file_path_display}\n\n> Source Content: {file_content}")
                    processed_files_count += 1

    except (IOError, OSError) as e:
        logger.error(f"Error walking or reading directory structure: {e}", exc_info=True)
        st.error(f"Error walking or reading directory structure: {e}")
        # Return potentially partial results
        return directory_structure.getvalue(), file_contents.getvalue()
    except Exception as e:
        logger.error(f"Unexpected error during directory processing: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during directory processing: {e}")
        return directory_structure.getvalue(), file_contents.getvalue()


    if processed_files_count == 0:
        logger.warning(f"No files matching allowed extensions {allowed_extensions} found.")
        st.warning(f"No files matching the allowed extensions ({', '.join(allowed_extensions)}) were found or processed in the ZIP(s).")
        if encountered_extensions:
             st.info(f"Found files with these extensions: {', '.join(sorted(list(encountered_extensions)))}. Consider adding them to 'Allowed Extensions'.")


    return directory_structure.getvalue(), file_contents.getvalue()

# --- Project Management ---

def load_projects() -> Dict[str, Dict[str, Any]]:
    """Loads project configurations from the JSON file."""
    if not os.path.exists(PROJECT_DATA_FILE):
        logger.info(f"Project file '{PROJECT_DATA_FILE}' not found. Initializing empty projects.")
        return {}
    try:
        with open(PROJECT_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                 logger.warning(f"Data in '{PROJECT_DATA_FILE}' is not a dictionary. Resetting.")
                 st.warning(f"Project file '{PROJECT_DATA_FILE}' has invalid format. Starting fresh.")
                 return {}
            logger.info(f"Loaded {len(data)} projects from '{PROJECT_DATA_FILE}'.")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{PROJECT_DATA_FILE}': {e}. Starting fresh.")
        st.error(f"Error loading projects from '{PROJECT_DATA_FILE}': Invalid JSON. Starting fresh.")
        # Optionally backup the corrupted file here
        return {}
    except (IOError, OSError) as e:
        logger.error(f"Error reading project file '{PROJECT_DATA_FILE}': {e}. Starting fresh.")
        st.error(f"Error loading projects from '{PROJECT_DATA_FILE}': {e}. Starting fresh.")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading projects: {e}", exc_info=True)
        st.error(f"An unexpected error occurred loading projects. Starting fresh.")
        return {}


def save_projects(projects: Dict[str, Dict[str, Any]]) -> bool:
    """Saves project configurations to the JSON file. Returns True on success."""
    try:
        with open(PROJECT_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(projects, f, indent=4)
        logger.info(f"Saved {len(projects)} projects to '{PROJECT_DATA_FILE}'.")
        return True
    except (IOError, OSError) as e:
        logger.error(f"Error saving projects to '{PROJECT_DATA_FILE}': {e}")
        st.error(f"Error saving projects to '{PROJECT_DATA_FILE}': {e}")
        return False
    except TypeError as e:
        logger.error(f"Error serializing project data to JSON: {e}", exc_info=True)
        st.error(f"Error saving projects: Data could not be serialized. {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving projects: {e}", exc_info=True)
        st.error(f"An unexpected error occurred saving projects.")
        return False


# --- State Management ---

def initialize_session_state():
    """Initializes Streamlit session state variables if they don't exist."""
    if 'projects_loaded' not in st.session_state:
        st.session_state.projects = load_projects()
        st.session_state.projects_loaded = True

    state_defaults = {
        # projects loaded above
        'current_project_name': lambda: list(st.session_state.projects.keys())[0] if st.session_state.projects else None,
        'generated_output': None,
        'output_filename': None,
        'estimated_token_count': None,
        'detected_extensions': None, # ADDED: To store extensions found in ZIP
        'last_uploaded_zip_names_for_scan': None, # ADDED: Track which files were scanned
    }
    for key, default_factory in state_defaults.items():
        if key not in st.session_state:
            # Use function call for dynamic defaults like loading projects
            st.session_state[key] = default_factory() if callable(default_factory) else default_factory

# --- UI Components ---

def render_sidebar():
    """Renders the sidebar for project management."""
    with st.sidebar:
        st.header("Project Management")

        project_names = [""] + sorted(list(st.session_state.projects.keys()))
        current_project = st.session_state.get('current_project_name')
        current_index = 0
        if current_project and current_project in project_names:
            try:
                current_index = project_names.index(current_project)
            except ValueError:
                 logger.warning(f"Current project '{current_project}' not found in project list. Resetting.")
                 st.session_state.current_project_name = None # Reset if inconsistent
                 current_index = 0


        def on_project_select():
            """Callback to handle project selection changes."""
            selected = st.session_state.selected_project_key
            logger.info(f"Project selected: {selected}")
            st.session_state.current_project_name = selected
            # Clear dependent state variables
            st.session_state.generated_output = None
            st.session_state.output_filename = None
            st.session_state.estimated_token_count = None
            st.session_state.detected_extensions = None # ADDED: Reset detected extensions
            st.session_state.last_uploaded_zip_names_for_scan = None # ADDED: Reset scan tracker
            # Clear uploader state if needed (Streamlit often handles this, but can be explicit)
            # uploader_key = f"uploader_{st.session_state.current_project_name}"
            # if uploader_key in st.session_state:
            #     st.session_state[uploader_key] = [] # Reset file uploader

        st.selectbox(
            "Select Project:",
            project_names,
            index=current_index,
            key="selected_project_key",
            on_change=on_project_select,
            format_func=lambda x: "Select a Project..." if not x else x
        )

        st.divider()
        render_create_project_section()
        st.divider()
        render_delete_project_section()
        st.divider()
        render_instructions()

def render_create_project_section():
    """Renders the 'Create New Project' UI in the sidebar."""
    st.subheader("Create New Project")
    new_project_name = st.text_input("New Project Name:", key="new_project_name_input")
    if st.button("Create Project"):
        handle_create_project(new_project_name)

def render_delete_project_section():
    """Renders the 'Delete Project' UI in the sidebar."""
    st.subheader("Delete Project")
    current_project = st.session_state.get('current_project_name')
    if current_project:
        # Add confirmation
        confirm_key = f"delete_confirm_{current_project}"
        delete_button_key = f"delete_button_{current_project}"

        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False

        if st.session_state[confirm_key]:
            st.warning(f"Are you sure you want to delete '{current_project}'?")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("YES, DELETE", key=f"confirm_delete_{current_project}"):
                    handle_delete_project(current_project)
                    st.session_state[confirm_key] = False # Reset confirmation state
                    st.rerun() # Rerun to update UI
            with col_cancel:
                if st.button("Cancel", key=f"cancel_delete_{current_project}"):
                    st.session_state[confirm_key] = False # Reset confirmation state
                    st.rerun() # Rerun to update UI
        else:
             if st.button(f"Delete Project '{current_project}'", key=delete_button_key):
                 st.session_state[confirm_key] = True
                 st.rerun() # Rerun to show confirmation

    else:
        st.info("Select a project to delete.")


def render_instructions():
    """Renders the instructions text."""
    st.info("""
        **Instructions:**
        1. Select or Create a project.
        2. Configure settings (Allowed Extensions, Output Pattern).
        3. **ZIP** the folder(s) containing your source files.
        4. Upload one or more `.zip` files.
        5. **Optional:** Click 'Scan Extensions' to see file types in the ZIP(s) and easily update 'Allowed Extensions'.
        6. Click 'Generate Prompt File'.
        7. Download the generated file.
    """)

def render_project_settings(project_name: str, project_data: Dict[str, Any]):
    """Renders the project settings inputs."""
    st.subheader("Project Settings")

    # Key for the text input widget itself
    extensions_input_key = f"ext_{project_name}"
    # --- *** REMOVE TEMP KEY *** ---
    # temp_update_key = f"temp_ext_update_{project_name}" # <-- REMOVED

    col1, col2 = st.columns(2)
    with col1:
        # --- *** START CHANGE *** ---
        # Simplify: Always determine the display value directly from the project data
        # (which might have been *just* updated by handle_save_settings before a rerun)
        display_ext_value = ", ".join(project_data.get("extensions", DEFAULT_EXTENSIONS))
        # --- *** END CHANGE *** ---

        # The value displayed in the text_input now comes from display_ext_value
        ext_str = st.text_input(
            "Allowed Extensions:",
            value=display_ext_value, # <-- Use the determined display value
            key=extensions_input_key,
            help="Comma-separated, lowercase, starting with '.', e.g., .md, .txt, .py"
        )
    with col2:
        # (Output pattern logic remains the same)
        default_pattern = DEFAULT_OUTPUT_PATTERN.format(project_name=project_name)
        current_pattern = project_data.get("output_filename_pattern", default_pattern)
        if "{project_name}" not in current_pattern:
             current_pattern = default_pattern

        out_pattern = st.text_input(
            "Output Filename Pattern:",
            value=current_pattern,
            key=f"out_{project_name}",
            help="Use {project_name} as a placeholder."
        )

    # This button is now less critical if the other button saves, but keep it for manual edits.
    if st.button("Save Project Settings"):
        # Pass the current value FROM THE TEXT INPUT WIDGET STATE
        handle_save_settings(project_name, project_data, st.session_state[extensions_input_key], st.session_state[f"out_{project_name}"])


def render_last_upload_info(project_data: Dict[str, Any]):
    """Displays info about the last processed ZIP file(s)."""
    last_zips = project_data.get("last_zip_names") # Expects a list
    last_time = project_data.get("last_processed_timestamp")
    if last_zips and isinstance(last_zips, list) and last_time:
        zip_display = ", ".join(f"**{name}**" for name in last_zips)
        st.caption(f":information_source: Last generated using: {zip_display} on {last_time}")

# --- NEW FUNCTION ---
def scan_zip_extensions(uploaded_zips: List[IO[bytes]]) -> Optional[List[str]]:
    """Scans uploaded zip files to find unique file extensions."""
    if not uploaded_zips:
        st.warning("Please upload at least one ZIP file to scan.")
        return None

    detected_extensions: Set[str] = set()
    processed_zip_names = []
    scan_errors = []

    with st.spinner("Scanning ZIP(s) for extensions..."):
        for zip_file_obj in uploaded_zips:
            zip_file_obj.seek(0) # Ensure reading from the start
            try:
                # Use ZipFile with the file-like object
                with zipfile.ZipFile(zip_file_obj, 'r') as zip_ref:
                    processed_zip_names.append(zip_file_obj.name)
                    for filename in zip_ref.namelist():
                         # Ignore directory entries and common macOS metadata folders/files
                        if not filename.endswith('/') and '__MACOSX/' not in filename and '.DS_Store' not in filename:
                            _, ext = os.path.splitext(filename)
                            if ext: # Only add if there is an extension
                                detected_extensions.add(ext.lower())
            except zipfile.BadZipFile:
                logger.warning(f"Skipping invalid/corrupted ZIP: {zip_file_obj.name}")
                scan_errors.append(f"Skipped invalid/corrupted ZIP: {zip_file_obj.name}")
            except Exception as e:
                logger.error(f"Error scanning ZIP {zip_file_obj.name}: {e}", exc_info=True)
                scan_errors.append(f"Error scanning {zip_file_obj.name}: {e}")
            finally:
                 zip_file_obj.seek(0) # Reset pointer for potential reuse

    if scan_errors:
         st.warning("Issues during ZIP scanning:\n" + "\n".join(scan_errors))

    if not detected_extensions and not scan_errors:
         st.info(f"No files with extensions found in the scanned ZIP(s): {', '.join(processed_zip_names)}")
         return [] # Return empty list if nothing found
    elif not detected_extensions and scan_errors:
         st.error("Scanning failed or no extensions found.")
         return None # Indicate failure or no results amidst errors


    sorted_extensions = sorted(list(detected_extensions))
    logger.info(f"Detected extensions in {', '.join(processed_zip_names)}: {sorted_extensions}")
    st.success(f"Found {len(sorted_extensions)} unique extension(s) in {', '.join(processed_zip_names)}.")
    return sorted_extensions

# --- MODIFIED FUNCTION ---
def render_zip_uploader_and_process(project_name: str, project_data: Dict[str, Any]):
    """Renders the ZIP uploader, extension scanner, and the 'Generate' button."""
    st.subheader("Upload & Process Source Code")

    # --- File Uploader ---
    uploader_key = f"uploader_{project_name}"
    uploaded_zips = st.file_uploader(
        "1. Select one or more .zip files:",
        type=["zip"],
        accept_multiple_files=True,
        key=uploader_key,
        help="Upload the zipped source code directories."
    )

    # --- Extension Scanning Section (Conditional) ---
    if uploaded_zips:
        current_uploaded_zip_names = sorted([f.name for f in uploaded_zips])

        col_scan1, col_scan2 = st.columns([1, 3]) # Adjust ratio as needed
        with col_scan1:
            if st.button("2. Scan Extensions", key=f"scan_ext_{project_name}", help="Scan uploaded ZIP(s) to see contained file extensions."):
                # Store the names of the files *at the time of scanning*
                st.session_state.last_uploaded_zip_names_for_scan = current_uploaded_zip_names
                # Perform the scan
                found_extensions = scan_zip_extensions(uploaded_zips)
                # Store results in session state (even if None or empty)
                st.session_state.detected_extensions = found_extensions
                # No explicit rerun needed here, state change triggers it

        # Display multiselect only if extensions were detected *and* the uploaded files haven't changed since last scan
        # Check if detected_extensions exists and is not None
        # Also check if the currently uploaded files match the ones that were scanned
        if st.session_state.get('detected_extensions') is not None and \
           st.session_state.get('last_uploaded_zip_names_for_scan') == current_uploaded_zip_names:

            detected_ext_list = st.session_state.detected_extensions
            if not detected_ext_list: # Handle case where scan found nothing
                 with col_scan2: # Show message in the second column
                     st.info("Scan complete. No file extensions were found in the uploaded ZIP(s).")
            else:
                # Get current allowed extensions from the project data for default selection
                current_allowed_ext_list = project_data.get("extensions", [])
                default_selection = [ext for ext in detected_ext_list if ext in current_allowed_ext_list]

                with st.container(): # Group multiselect and button
                     st.markdown("---") # Visual separator
                     st.markdown("**Detected Extensions:** (Select to update field below)")

                     cols_multi = st.columns([3, 1]) # Column for multiselect, column for button
                     with cols_multi[0]:
                         selected_extensions = st.multiselect(
                            label="Select extensions to include in 'Allowed Extensions':", # Use label_visibility="collapsed" if markdown is enough
                            label_visibility="collapsed",
                            options=detected_ext_list,
                            default=default_selection,
                            key=f"detected_extensions_multiselect_{project_name}"
                         )
                     with cols_multi[1]:
                          # Define the key for the output pattern input
                          output_pattern_key = f"out_{project_name}"

                          # --- *** START CHANGE *** ---
                          # Change button text, key, and help text
                          if st.button("Use Selected & Save Settings",
                                       key=f"update_and_save_ext_{project_name}",
                                       help="Update 'Allowed Extensions' with the selection and save all project settings."):

                              # 1. Get selected extensions from multiselect
                              new_ext_string = ", ".join(selected_extensions)

                              # 2. Get the CURRENT value from the Output Filename Pattern input
                              #    It's crucial to read the current state of that widget
                              if output_pattern_key in st.session_state:
                                   current_output_pattern = st.session_state[output_pattern_key]
                              else:
                                   # Fallback if the key isn't initialized yet (shouldn't happen in normal flow)
                                   current_output_pattern = project_data.get("output_filename_pattern",
                                                                            DEFAULT_OUTPUT_PATTERN.format(project_name=project_name))
                                   logger.warning(f"Output pattern key '{output_pattern_key}' not found in session state. Using fallback.")


                              # 3. Call the existing save handler directly
                              logger.info(f"Attempting to save settings for '{project_name}' via 'Use Selected & Save' button.")
                              logger.info(f"  - Extensions to save: '{new_ext_string}'")
                              logger.info(f"  - Output pattern to save: '{current_output_pattern}'")
                              handle_save_settings(project_name, project_data, new_ext_string, current_output_pattern)

                              # 4. Remove info message about needing to save separately
                              #    Success/Error messages are handled within handle_save_settings

                              # 5. Optional: Rerun might still be good practice ensure UI consistency,
                              #    though handle_save_settings might implicitly cause necessary updates
                              #    if it modifies project_data successfully. Test if needed.
                              st.rerun() # Keep rerun for now to ensure text input updates reliably

                          # --- *** END CHANGE *** ---
                     st.markdown("---") # Visual separator

    # --- Generate Button ---
    if uploaded_zips:
        # Retrieve the currently configured extensions from project data for the generation step
        # This ensures we use the SAVED settings, not just what's visually in the text box
        try:
             # Use the saved project data for allowed extensions
             allowed_extensions_list = project_data.get("extensions", [])
             if not allowed_extensions_list:
                 st.error("Project has no 'Allowed Extensions' configured. Please set and save them first.")
                 # Disable button indirectly by not rendering it or using st.button(..., disabled=True)
                 generate_disabled = True
             else:
                 allowed_extensions_tuple = tuple(allowed_extensions_list)
                 generate_disabled = False

        except Exception as e:
            logger.error(f"Error retrieving allowed extensions for project {project_name}: {e}")
            st.error("Could not retrieve project settings. Cannot generate.")
            generate_disabled = True

        # Button to generate the final prompt file
        if st.button("3. Generate Prompt File from ZIP(s)", key=f"generate_{project_name}", disabled=generate_disabled, type="primary"):
             if not generate_disabled:
                 handle_generate_prompt(project_name, project_data, uploaded_zips, allowed_extensions_tuple)
             else:
                  st.warning("Generation disabled due to missing configuration.")

    else:
        st.info("Upload ZIP file(s) to enable scanning and generation.")


def render_results_section():
    """Renders the token count and download button if output exists."""
    generated_output = st.session_state.get('generated_output')
    token_count = st.session_state.get('estimated_token_count')
    output_filename = st.session_state.get('output_filename') # Get filename from state

    if generated_output is not None: # Check explicitly for None (empty string is valid output)
        st.divider()
        st.subheader("Results")
        if token_count is not None: # Check for None explicitly
            # Ensure token_count is int for formatting
            try:
                 token_count_int = int(token_count)
                 st.info(f"Estimated Token Count: **{token_count_int:,}** (~{len(generated_output):,} chars / 4)")
            except (ValueError, TypeError):
                 st.info(f"Estimated Token Count: N/A (Could not calculate)")
                 logger.warning(f"Could not convert estimated token count '{token_count}' to int.")
        else:
            st.info("Token count could not be estimated.")

        if output_filename:
            try:
                # Ensure output is bytes for download
                output_bytes = generated_output.encode('utf-8')
                st.download_button(
                    label=f"Download {output_filename}",
                    data=output_bytes,
                    file_name=output_filename,
                    mime='text/plain',
                    key='download_btn'
                )
                # Add copy-to-clipboard button here if desired
                # st.code(generated_output, language=None) # Display content if needed
                # st.button("Copy to Clipboard", on_click=lambda: pyperclip.copy(generated_output)) # Needs pyperclip install

            except Exception as e:
                logger.error(f"Error preparing download button: {e}", exc_info=True)
                st.error(f"Error preparing download button: {e}")
        else:
            st.warning("Output generated, but filename is missing.")
            logger.warning("Generated output exists, but output_filename is not set in session state.")


def render_main_content():
    """Renders the main content area based on the selected project."""
    current_project_name = st.session_state.get('current_project_name')

    if not current_project_name:
        st.info("Please select or create a project from the sidebar.")
        return

    st.header(f"Project: {current_project_name}")
    project_data = st.session_state.projects.get(current_project_name)

    if not project_data:
        st.error(f"Data for project '{current_project_name}' not found. Please select again or create it.")
        # Attempt to reset selection
        st.session_state.current_project_name = None
        st.session_state.selected_project_key = "" # Reset selectbox widget state if possible/needed
        logger.error(f"Project data inconsistency for '{current_project_name}'. Forcing project selection reset.")
        st.rerun()
        return

    render_project_settings(current_project_name, project_data)
    st.divider()
    render_last_upload_info(project_data) # Show info about last *generation*
    render_zip_uploader_and_process(current_project_name, project_data)
    render_results_section() # Render results including token count and download

# --- Action Handlers ---

def handle_create_project(project_name: str):
    """Handles creating a new project."""
    project_name = project_name.strip()
    if not project_name:
        st.warning("Please enter a project name.")
        return
    # Add validation for invalid characters if needed (e.g., '/', '\')
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in project_name for char in invalid_chars):
         st.error(f"Project name contains invalid characters: {', '.join(invalid_chars)}")
         return

    if project_name in st.session_state.projects:
        st.warning(f"Project '{project_name}' already exists.")
        return

    st.session_state.projects[project_name] = {
        "extensions": list(DEFAULT_EXTENSIONS), # Ensure it's a list
        "output_filename_pattern": DEFAULT_OUTPUT_PATTERN.format(project_name=project_name),
        "last_zip_names": None, # Use list for consistency, even if None
        "last_processed_timestamp": None
    }
    if save_projects(st.session_state.projects):
        st.session_state.current_project_name = project_name
        logger.info(f"Project '{project_name}' created successfully.")
        st.success(f"Project '{project_name}' created!")
        # Clear potentially stale state from previous project/no project
        st.session_state.detected_extensions = None
        st.session_state.last_uploaded_zip_names_for_scan = None
        st.rerun() # Rerun to reflect changes
    else:
        st.session_state.projects.pop(project_name, None) # Revert if save failed
        logger.error(f"Failed to save project '{project_name}' after creation.")


def handle_delete_project(project_name: str):
    """Handles deleting a project."""
    if project_name not in st.session_state.projects:
        st.error("Project not found for deletion.")
        logger.warning(f"Attempted to delete non-existent project: {project_name}")
        return

    del st.session_state.projects[project_name]
    if save_projects(st.session_state.projects):
        st.success(f"Project '{project_name}' deleted.")
        logger.info(f"Project '{project_name}' deleted successfully.")
        st.session_state.current_project_name = None
        # Clear potentially stale state
        st.session_state.detected_extensions = None
        st.session_state.last_uploaded_zip_names_for_scan = None
        # No rerun needed here, handled by button logic/confirmation reset
    else:
        # Reload projects from file to revert the deletion attempt if save failed
        st.session_state.projects = load_projects()
        logger.error(f"Failed to save projects after deleting '{project_name}'. Reverted deletion.")
        st.error(f"Failed to save changes after deleting '{project_name}'. Deletion reverted.")
        # Rerun might be needed to refresh UI if reload changed things
        st.rerun()


def validate_project_settings(ext_str: str, output_pattern: str) -> Optional[Tuple[List[str], str]]:
    """Validates settings. Returns (valid_extensions_list, cleaned_pattern_str) or None if invalid."""
    valid_extensions = []
    has_error = False
    error_messages = []

    # Validate Extensions
    if ext_str:
        # Split, strip whitespace, filter empty strings, convert to lowercase
        extensions = [e.strip().lower() for e in ext_str.split(',') if e.strip()]
        if not extensions:
            error_messages.append("Allowed extensions were provided but resulted in an empty list (e.g., just commas?). Please provide valid extensions like '.py, .md'.")
            has_error = True
        else:
            valid_extensions_set = set() # Use set to handle duplicates
            for ext in extensions:
                if not ext.startswith('.'):
                    error_messages.append(f"Invalid extension format: '{ext}'. Extensions must start with '.'")
                    has_error = True
                elif len(ext) == 1: # Just "."
                    error_messages.append(f"Invalid extension format: '{ext}'. Extension cannot be just a dot.")
                    has_error = True
                elif ' ' in ext:
                    error_messages.append(f"Invalid extension format: '{ext}'. Extensions cannot contain spaces.")
                    has_error = True
                else:
                    valid_extensions_set.add(ext)
            valid_extensions = sorted(list(valid_extensions_set)) # Sort for consistency
    else:
        # Allow empty list if user explicitly clears it? Decide policy.
        # Current policy: Treat empty string as valid, resulting in empty list.
        # If it should be an error:
        # error_messages.append("Allowed extensions cannot be empty. Use defaults or specify extensions like '.txt'.")
        # has_error = True
        valid_extensions = [] # Explicitly empty list if input is empty/whitespace
        st.warning("Allowed Extensions field is empty. No files will be processed unless extensions are added.")


    # Validate Output Pattern
    output_pattern_cleaned = output_pattern.strip()
    if not output_pattern_cleaned:
        error_messages.append("Output Filename Pattern cannot be empty.")
        has_error = True
    # Add more validation for invalid filename chars if needed
    invalid_filename_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    # Check pattern excluding the placeholder itself
    pattern_test_name = "test_project"
    try:
         test_filename = output_pattern_cleaned.format(project_name=pattern_test_name)
         if any(char in test_filename for char in invalid_filename_chars):
             error_messages.append(f"Output Filename Pattern results in invalid characters: {', '.join(invalid_filename_chars)}")
             has_error = True
    except KeyError:
         # This should be caught by the {project_name} check above, but good practice
         error_messages.append("Output Filename Pattern has incorrect formatting (e.g., mismatched braces).")
         has_error = True
    except Exception as e:
         error_messages.append(f"Unexpected error validating filename pattern: {e}")
         has_error = True


    if has_error:
        for msg in error_messages:
            st.error(msg)
        logger.warning(f"Project settings validation failed: {error_messages}")
        return None
    else:
        return valid_extensions, output_pattern_cleaned

def handle_save_settings(project_name: str, project_data: Dict[str, Any], ext_str: str, output_pattern: str):
    """Validates and saves project settings."""
    validation_result = validate_project_settings(ext_str, output_pattern)
    if validation_result is None:
        # Errors already shown by validation function
        return

    valid_extensions, cleaned_pattern = validation_result

    # Update the project_data dictionary (which is a reference to the one in st.session_state.projects)
    project_data["extensions"] = valid_extensions
    project_data["output_filename_pattern"] = cleaned_pattern

    # No need to reassign: st.session_state.projects[project_name] = project_data
    # because project_data is already the dictionary object within st.session_state.projects

    if save_projects(st.session_state.projects):
        st.success("Project settings saved!")
        logger.info(f"Saved settings for project '{project_name}'. Extensions: {valid_extensions}, Pattern: '{cleaned_pattern}'")
        # Optional: Rerun if other parts of the UI depend critically on the saved state immediately
        # st.rerun()
    else:
        # Attempt to reload projects to revert the failed save in memory
        st.error("Failed to save project settings to disk. Changes may be lost on reload.")
        logger.error(f"Failed to save project settings for '{project_name}' to disk.")
        st.session_state.projects = load_projects() # Reload to ensure consistency with saved state


def estimate_token_count(text: Optional[str]) -> Optional[int]:
    """Roughly estimates token count (GPT-like models). 1 token ~ 4 chars."""
    if text is None:
        return None
    if not isinstance(text, str):
         logger.warning(f"Estimate token count received non-string input: {type(text)}")
         return None
    # This is a very rough approximation, especially for code or non-English text
    return len(text) // 4

def handle_generate_prompt(
    project_name: str,
    project_data: Dict[str, Any],
    uploaded_zips: List[IO[bytes]],
    allowed_extensions: Tuple[str, ...] # Pass the validated, saved extensions
):
    """Handles ZIP extraction, processing, and updating state."""
    # Pre-checks moved to render_zip_uploader_and_process to disable button
    if not allowed_extensions:
        st.error("Cannot process: No allowed extensions configured in saved settings.")
        logger.error(f"Generate cancelled for '{project_name}': No allowed extensions.")
        return
    if not uploaded_zips:
        st.warning("No ZIP files are currently uploaded.")
        logger.warning(f"Generate cancelled for '{project_name}': No files uploaded.")
        return

    processed_successfully = False
    final_output = None
    structure = ""
    content = ""
    zip_names = [zf.name for zf in uploaded_zips]

    # Reset previous results before starting
    st.session_state.generated_output = None
    st.session_state.estimated_token_count = None
    st.session_state.output_filename = None


    with st.spinner(f"Processing {len(uploaded_zips)} ZIP file(s)..."):
        logger.info(f"Starting generation for project '{project_name}' with {len(zip_names)} ZIP(s): {', '.join(zip_names)}")
        logger.info(f"Using allowed extensions: {allowed_extensions}")
        try:
            # Use a temporary directory that cleans itself up
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Created temporary directory: {temp_dir}")
                any_extraction_ok = False
                extraction_errors = []
                for zip_file in uploaded_zips:
                    zip_file.seek(0) # Go to start of file stream
                    try:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                             # Safe extraction (consider adding checks for zip bombs if needed)
                             zip_ref.extractall(temp_dir)
                        logger.info(f"Successfully extracted '{zip_file.name}' to {temp_dir}")
                        any_extraction_ok = True
                    except zipfile.BadZipFile:
                        msg = f"Skipped invalid/corrupted ZIP: {zip_file.name}"
                        extraction_errors.append(msg)
                        logger.warning(msg)
                    except Exception as e:
                        msg = f"Error extracting {zip_file.name}: {e}"
                        extraction_errors.append(msg)
                        logger.error(msg, exc_info=True)
                    finally:
                         zip_file.seek(0) # Reset pointer again

                if extraction_errors:
                    st.warning("Issues during ZIP extraction:\n" + "\n".join(extraction_errors))

                if not any_extraction_ok:
                    st.error("Extraction failed for all uploaded ZIP files.")
                    logger.error(f"Extraction failed for all ZIPs in project '{project_name}'.")
                    # No need to return here, let it flow to update state below
                else:
                    # --- Process the extracted content ---
                    try:
                        structure, content = get_structure_and_content_from_dir(
                            temp_dir, allowed_extensions
                        )
                        # Check if *anything* was generated (even if no matching files found)
                        if structure is not None and content is not None:
                             final_output = (
                                f"--- Project: {project_name} ---\n\n"
                                f"--- Source Structure (from {', '.join(zip_names)}) ---\n\n{structure.strip()}\n\n"
                                f"--- Source Content ---\n{content.strip()}"
                             )
                             processed_successfully = True # Mark success if processing ran without error
                             logger.info(f"Successfully processed content for project '{project_name}'. Structure length: {len(structure)}, Content length: {len(content)}")
                        else:
                             # Should not happen if get_structure_and_content_from_dir always returns strings
                             logger.error(f"Content processing returned None for project '{project_name}'.")
                             st.error("Internal error: Failed to get processed content.")


                    except Exception as e:
                         st.error(f"Error during content processing after extraction: {e}")
                         logger.error(f"Error during content processing for '{project_name}': {e}", exc_info=True)
                         # Keep processed_successfully as False

        except Exception as e:
            st.error(f"Failed processing ZIPs (e.g., temporary directory issue): {e}")
            logger.error(f"Failed processing ZIPs for '{project_name}' (temp dir issue?): {e}", exc_info=True)
            # Keep processed_successfully as False

    # --- Update state post-processing ---
    st.session_state.generated_output = final_output # Will be None if processing failed or yielded nothing
    st.session_state.estimated_token_count = estimate_token_count(final_output)

    if processed_successfully and final_output is not None:
        st.success("Processing complete! Results are available below.")
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Update project data with info about this successful generation run
            project_data["last_zip_names"] = zip_names
            project_data["last_processed_timestamp"] = current_time
            # Update the project data in the main session state
            st.session_state.projects[project_name] = project_data

            # Determine the output filename using the saved pattern
            try:
                 output_filename = project_data.get("output_filename_pattern", DEFAULT_OUTPUT_PATTERN).format(project_name=project_name)
                 st.session_state.output_filename = output_filename
            except KeyError:
                 st.error("Error formatting output filename: '{project_name}' placeholder missing or invalid pattern.")
                 logger.error(f"Invalid output filename pattern for {project_name}: {project_data.get('output_filename_pattern')}")
                 st.session_state.output_filename = f"{project_name}_Prompt_Fallback.txt" # Fallback filename
            except Exception as e:
                 st.error(f"Error generating output filename: {e}")
                 logger.error(f"Error formatting output filename for {project_name}: {e}", exc_info=True)
                 st.session_state.output_filename = f"{project_name}_Prompt_Fallback.txt" # Fallback filename

            # Save the updated project data (last processed info)
            if save_projects(st.session_state.projects):
                logger.info(f"Successfully updated last processed info for project '{project_name}'.")
            else:
                 st.warning("Could not save the update for 'last processed file' information.")
                 logger.warning(f"Failed to save project data after successful generation for '{project_name}'.")

            # No explicit rerun needed here, state changes trigger update for results section
            # st.rerun() # Avoid rerun if possible to prevent resetting inputs unnecessarily

        except Exception as e:
            st.warning(f"Error occurred while updating project metadata after generation: {e}")
            logger.error(f"Error updating project metadata for '{project_name}' after generation: {e}", exc_info=True)

    elif final_output is None and any_extraction_ok:
        # Processing happened, but yielded no output (e.g., no matching files found)
        # Warnings/info messages should have been displayed by get_structure_and_content_from_dir
        logger.info(f"Processing completed for '{project_name}', but no content was generated (likely no matching files).")
        # Ensure results section knows there's nothing to download
        st.session_state.generated_output = None
        st.session_state.output_filename = None
        st.session_state.estimated_token_count = None


    elif not any_extraction_ok:
         # Extraction failed completely
         logger.error(f"Generation failed for '{project_name}' due to extraction errors.")
         # State already cleared/set to None at start of function


# --- Main Application Flow ---

def main():
    """Main function to set up and run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="Project Prompt Generator")
    st.title("Project Content Processor & Prompt Generator")

    # Initialize state once at the beginning
    initialize_session_state()

    # Render UI components
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()