import streamlit as st
import yaml
import subprocess
import shutil
import requests
import os
import logging
import sys
import urllib.parse
import tarfile
from pathlib import Path
from google import genai
from google.genai import types

# Page Config
st.set_page_config(
    page_title="Refactored OpenAPI Validator (v10)",
    page_icon="📘",
    layout="wide"
)

# --- CRITICAL FIX: Manually Install Node.js v20 ---
def ensure_node_installed():
    """
    Streamlit Cloud's default apt-get installs ancient Node.js (v10/v12).
    rdme requires Node v20+.
    This function manually downloads a standalone Node v20 binary and adds it to PATH.
    """
    node_version = "v20.11.0"
    install_dir = Path("./node_runtime")
    node_bin_path = install_dir / f"node-{node_version-linux-x64}" / "bin"
    
    # Check if we already have the right node version
    try:
        # Check if 'node' is in path and get version
        result = subprocess.run(["node", "-v"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip().startswith("v20"):
            return # Already good!
    except FileNotFoundError:
        pass

    # If not found or old, install it locally
    if not (install_dir / "bin" / "node").exists():
        with st.spinner(f"🔧 Installing Node.js {node_version} (Required for ReadMe v10)..."):
            url = f"https://nodejs.org/dist/{node_version}/node-{node_version}-linux-x64.tar.xz"
            
            # Download
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                tar_path = Path("node.tar.xz")
                with open(tar_path, 'wb') as f:
                    f.write(response.raw.read())
                
                # Extract
                with tarfile.open(tar_path) as tar:
                    tar.extractall(install_dir)
                
                # Cleanup
                os.remove(tar_path)
            else:
                st.error("Failed to download Node.js runtime.")
                st.stop()
    
    # Add to PATH for this process
    # The extraction creates a folder like 'node-v20.11.0-linux-x64/bin'
    extracted_folder = list(install_dir.glob("node-v*-linux-x64"))[0]
    bin_path = extracted_folder / "bin"
    
    # Update Environment Variables for the current process
    os.environ["PATH"] = f"{str(bin_path.absolute())}{os.pathsep}{os.environ['PATH']}"
    
    # Verify
    try:
        ver = subprocess.check_output(["node", "-v"], text=True).strip()
        # st.toast(f"✅ Runtime Ready: {ver}") # Optional visual confirmation
    except Exception as e:
        st.error(f"Failed to set up Node runtime: {e}")

# --- Initialize Session State for Logs ---
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- Custom Logging Handler for Streamlit ---
class StreamlitLogHandler(logging.Handler):
    def __init__(self, container, download_placeholder=None):
        super().__init__()
        self.container = container
        self.download_placeholder = download_placeholder

    def emit(self, record):
        msg = self.format(record)
        st.session_state.logs.append(msg)
        self.container.code("\n".join(st.session_state.logs), language="text")
        
        if self.download_placeholder:
            unique_key = f"log_dl_rt_{len(st.session_state.logs)}"
            self.download_placeholder.download_button(
                label="📥 Download Log File",
                data="\n".join(st.session_state.logs),
                file_name="openapi_upload.log",
                mime="text/plain",
                key=unique_key
            )

# --- Helper Functions ---

def get_npx_path():
    # Because we updated os.environ["PATH"] in ensure_node_installed(),
    # shutil.which should now find the correct npx in our local folder.
    return shutil.which("npx")

def validate_env(api_key, required=True):
    if not api_key:
        if required:
            st.error("❌ ReadMe API Key is missing. Please enter it in the sidebar.")
            st.stop()
        return False
    return True

def run_command(command_list, log_logger):
    try:
        cmd_str = " ".join(command_list)
        log_logger.info(f"Running: {cmd_str}")
        
        # We must pass the updated os.environ to the subprocess
        # so it inherits the PATH with our new Node.js binary
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=os.environ.copy() 
        )
        for line in process.stdout:
            clean = line.strip()
            if clean:
                log_logger.info(f"[CLI] {clean}")
        process.wait()
        return process.returncode
    except Exception as e:
        log_logger.error(f"❌ Command failed: {e}")
        return 1

# --- AI Analysis Logic ---
def analyze_errors_with_ai(log_content, api_key, model_name):
    if not api_key: return None
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are an expert OpenAPI Validator. Analyze the following log output.
        Identify specific YAML errors and provide actionable solutions.
        Logs:
        {log_content}
        """
        response = client.models.generate_content(model=model_name, contents=[prompt])
        return response.text
    except Exception as e:
        return f"Exception calling AI: {e}"

# --- Git Logic ---
def setup_git_repo(repo_url, repo_dir, git_token, git_username, branch_name, logger):
    logger.info(f"🚀 Starting Git Operation for branch: {branch_name}...")
    repo_path = Path(repo_dir)
    
    if repo_url: repo_url = repo_url.strip().strip('"').strip("'")
    if git_username: git_username = git_username.strip().strip('"').strip("'")
    if git_token: git_token = git_token.strip().strip('"').strip("'")

    if repo_url and repo_url.count("https://") > 1:
        match = re.search(r"(https://github\.com/.*)$", repo_url)
        if match: repo_url = match.group(1)

    auth_repo_url = repo_url
    try:
        if repo_url and git_username and git_token:
            parsed = urllib.parse.urlparse(repo_url)
            safe_user = urllib.parse.quote(git_username, safe='')
            safe_token = urllib.parse.quote(git_token, safe='')
            clean_netloc = parsed.netloc.split("@")[-1] if "@" in parsed.netloc else parsed.netloc
            auth_netloc = f"{safe_user}:{safe_token}@{clean_netloc}"
            auth_repo_url = urllib.parse.urlunparse((parsed.scheme, auth_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    except Exception as e:
        logger.error(f"❌ URL Construction Failed: {e}")
        st.stop()

    env_vars = os.environ.copy()
    env_vars["GIT_TERMINAL_PROMPT"] = "0"
    
    if not repo_path.exists():
        logger.info(f"⬇️ Cloning branch '{branch_name}'...")
        cmd = ["git", "clone", "--depth", "1", "--branch", branch_name, auth_repo_url, str(repo_path)]
        if run_command(cmd, logger) != 0:
            st.error("Git Clone Failed.")
            st.stop()
    else:
        logger.info(f"🔄 Fetching latest for '{branch_name}'...")
        try:
            subprocess.run(["git", "-C", str(repo_path), "remote", "set-url", "origin", auth_repo_url], check=True, env=env_vars)
            subprocess.run(["git", "-C", str(repo_path), "fetch", "origin", branch_name], check=True, env=env_vars)
            subprocess.run(["git", "-C", str(repo_path), "reset", "--hard", f"origin/{branch_name}"], check=True, env=env_vars)
        except Exception:
            logger.warning("Git update failed, using existing files.")

def delete_repo(repo_dir):
    path = Path(repo_dir)
    if path.exists():
        shutil.rmtree(path)
        return True, "Deleted successfully."
    return False, "Path does not exist."

# --- File Operations ---
def prepare_files(filename, paths, workspace, dependency_list, logger):
    source = None
    main_candidate = Path(paths['specs']) / f"{filename}.yaml"
    if main_candidate.exists():
        source = main_candidate
    elif paths.get('secondary') and (Path(paths['secondary']) / f"{filename}.yaml").exists():
        source = Path(paths['secondary']) / f"{filename}.yaml"

    if not source:
        logger.error(f"❌ Source file '{filename}.yaml' not found.")
        st.stop()

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    destination = workspace_path / source.name
    shutil.copy(source, destination)
    logger.info(f"📂 Copied main YAML to workspace: {destination.name}")

    for folder in dependency_list:
        clean_folder = folder.strip()
        if not clean_folder: continue
        src_folder = Path(paths['specs']) / clean_folder
        dest_folder = workspace_path / clean_folder
        if src_folder.exists():
            if dest_folder.exists(): shutil.rmtree(dest_folder)
            shutil.copytree(src_folder, dest_folder)
            logger.info(f"📂 Copied dependency folder: {clean_folder}")
    return destination

def process_yaml_content(file_path, api_domain, logger):
    logger.info("🛠️ Injecting x-readme extensions...")
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        if "openapi" in data:
            pos = list(data.keys()).index("openapi")
            items = list(data.items())
            items.insert(pos + 1, ("x-readme", {"explorer-enabled": False}))
            data = dict(items)
        
        domain = api_domain if api_domain else "example.com"
        if "servers" not in data or not data["servers"]:
            data["servers"] = [{"url": f"https://{domain}", "variables": {}}]

        if "variables" not in data["servers"][0]:
            data["servers"][0]["variables"] = {}
            
        data["servers"][0]["variables"]["base-url"] = {"default": domain}
        data["servers"][0]["variables"]["protocol"] = {"default": "https"}

        edited_path = file_path.parent / (file_path.stem + "_edited.yaml")
        with open(edited_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        return edited_path
    except Exception as e:
        logger.error(f"❌ Error processing YAML: {e}")
        st.stop()

# --- CALLBACKS ---
def clear_credentials():
    st.session_state.readme_key = ""
    st.session_state.git_user = ""
    st.session_state.git_token = ""
    st.session_state.logs = []

def clear_logs():
    st.session_state.logs = []

# --- MAIN ---
def main():
    # 1. ENSURE NODE v20 IS PRESENT
    ensure_node_installed()

    st.sidebar.title("⚙️ Refactored Config")
    if 'readme_key' not in st.session_state: st.session_state.readme_key = ""
    if 'gemini_key' not in st.session_state: st.session_state.gemini_key = ""
    if 'git_user' not in st.session_state: st.session_state.git_user = ""
    if 'git_token' not in st.session_state: st.session_state.git_token = ""

    readme_key = st.sidebar.text_input("ReadMe API Key", key="readme_key", type="password")
    
    with st.sidebar.expander("🤖 AI Configuration"):
        gemini_key = st.text_input("Gemini API Key", key="gemini_key", type="password")
        ai_model = st.text_input("Model", key="ai_model", value="gemini-2.0-flash")

    st.sidebar.subheader("Git Repo Config")
    repo_path = st.sidebar.text_input("Local Clone Path", value="./cloned_repo")
    if st.sidebar.button("🗑️ Reset Repo"):
        success, msg = delete_repo(repo_path)
        if success: st.sidebar.success(msg)

    repo_url = st.sidebar.text_input("Git Repo URL", key="repo_url")
    branch_name = st.sidebar.text_input("Branch Source", value="main")
    git_user = st.sidebar.text_input("Git User", key="git_user")
    git_token = st.sidebar.text_input("Git Token", key="git_token", type="password")
    st.sidebar.button("🔒 Clear Credentials", on_click=clear_credentials)

    st.sidebar.subheader("Internal Paths")
    spec_rel_path = st.sidebar.text_input("Main Specs Path", value="specs")
    secondary_rel_path = st.sidebar.text_input("Secondary Path", value="")
    dep_input = st.sidebar.text_input("Dependencies", value="common")
    dependency_list = [x.strip() for x in dep_input.split(",")]
    api_domain = st.sidebar.text_input("API Domain", value="api.example.com")

    abs_spec_path = Path(repo_path) / spec_rel_path
    paths = {"repo": repo_path, "specs": abs_spec_path}
    if secondary_rel_path:
        paths["secondary"] = Path(repo_path) / secondary_rel_path
    workspace_dir = "./temp_workspace"

    st.title("🚀 Refactored OpenAPI Validator")

    col1, col2 = st.columns(2)
    with col1:
        files = []
        if abs_spec_path.exists():
            files.extend([f.stem for f in abs_spec_path.glob("*.yaml")])
        if "secondary" in paths and paths["secondary"].exists():
            files.extend([f.stem for f in paths["secondary"].glob("*.yaml")])
        files = sorted(list(set(files)))
        
        if files:
            selected_file = st.selectbox("Select File", files)
        else:
            selected_file = st.text_input("Filename", "audit")

    with col2:
        target_branch = st.text_input("Target ReadMe Branch", "main")

    st.markdown("### Settings")
    c1, c2, c3 = st.columns(3)
    use_swagger = c1.checkbox("Swagger", True)
    use_redocly = c2.checkbox("Redocly", True)
    use_readme = c3.checkbox("ReadMe CLI", False)

    c_btn1, c_btn2 = st.columns(2)
    btn_validate = c_btn1.button("🔍 Validate")
    btn_upload = c_btn2.button("🚀 Upload", type="primary")

    log_container = st.empty()
    download_placeholder = st.empty()

    if btn_validate or btn_upload:
        st.session_state.logs = []
        logger = logging.getLogger("streamlit_logger")
        logger.setLevel(logging.INFO)
        if logger.handlers: logger.handlers = []
        handler = StreamlitLogHandler(log_container, download_placeholder)
        logger.addHandler(handler)

        has_key = validate_env(readme_key, required=btn_upload)
        npx_path = get_npx_path()
        if not npx_path:
            logger.error("❌ NodeJS/npx not found (even after manual install).")
            st.stop()

        setup_git_repo(repo_url, repo_path, git_token, git_user, branch_name, logger)
        final_yaml_path = prepare_files(selected_file, paths, workspace_dir, dependency_list, logger)
        edited_file = process_yaml_content(final_yaml_path, api_domain, logger)
        st.session_state.last_edited_file = str(edited_file)

        failed = False
        if use_swagger and run_command([npx_path, "--yes", "swagger-cli", "validate", str(edited_file)], logger) != 0: failed = True
        if use_redocly and run_command([npx_path, "--yes", "@redocly/cli@1.25.0", "lint", str(edited_file)], logger) != 0: failed = True
        
        if use_readme and has_key:
            if run_command([npx_path, "--yes", "rdme", "openapi:validate", str(edited_file)], logger) != 0: failed = True

        if failed:
            st.error("Validation Failed.")
            if btn_upload: st.stop()
        elif btn_upload:
            logger.info(f"🚀 Uploading to branch: {target_branch}")
            cmd = [npx_path, "--yes", "rdme", "openapi", "upload", str(edited_file), "--key", readme_key, "--branch", target_branch]
            if run_command(cmd, logger) == 0:
                st.success("✅ Uploaded successfully!")
            else:
                st.error("❌ Upload failed.")
        else:
            st.success("Validation Passed.")

    # Post-execution UI
    with st.expander("Downloads & Tools"):
        if 'last_edited_file' in st.session_state and st.session_state.last_edited_file:
            path = Path(st.session_state.last_edited_file)
            if path.exists():
                with open(path, "r") as f:
                    st.download_button("Download YAML", f.read(), path.name)
        if st.session_state.logs:
             st.button("Clear Logs", on_click=clear_logs)

    if st.session_state.logs and gemini_key:
        if st.button("Analyze Logs with AI"):
             with st.spinner("Analyzing..."):
                analysis = analyze_errors_with_ai("\n".join(st.session_state.logs), gemini_key, st.session_state.ai_model)
                st.markdown(analysis)

if __name__ == "__main__":
    main()
