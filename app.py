"""HuggingFace Spaces entry point — forwards to app/streamlit_app.py."""
# HF Spaces requires app.py at the repo root.
# We add the project root to sys.path so all src/ imports resolve,
# then re-execute the actual app module.
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))  # ensure relative paths (configs/, data/) work

exec(open(os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")).read())
