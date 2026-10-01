# ragfix for VS Code

Run ragfix from the editor, save your OpenAI key in VS Code secret storage, and inspect which retrieved chunks support each generated sentence.

## Install

1. Install the Python package:

```powershell
pip install ragfix
```

2. Install this extension from the VSIX (Command Palette: **Extensions: Install from VSIX...**), or:

```powershell
code --install-extension ragfix-0.1.0.vsix
```

In Cursor:

```powershell
cursor --install-extension ragfix-0.1.0.vsix
```

## Use

1. Command Palette: **RAG Debugger: Set OpenAI API Key**
2. **RAG Debugger: Start**
3. Upload PDF/TXT files and ask a question in the UI

You can also open the **RAG Debugger** icon in the activity bar.

The API key is stored with VS Code secret storage. It is not written into the git repo.
