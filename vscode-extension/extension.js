const vscode = require('vscode');
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

const SECRET_KEY = 'ragDebugger.openaiApiKey';

/** @type {import('child_process').ChildProcess | undefined} */
let serverProcess;
/** @type {SidebarProvider | undefined} */
let sidebar;
let statusBar;

function getConfig() {
  const cfg = vscode.workspace.getConfiguration('ragDebugger');
  return {
    host: cfg.get('host') || '127.0.0.1',
    port: Number(cfg.get('port') || 8000),
    pythonPath: (cfg.get('pythonPath') || '').trim(),
  };
}

function serverUrl() {
  const { host, port } = getConfig();
  return `http://${host}:${port}`;
}

function uiUrl() {
  return `${serverUrl()}/ui`;
}

function findProjectRoot(extensionPath) {
  const folders = vscode.workspace.workspaceFolders || [];
  for (const folder of folders) {
    if (fs.existsSync(path.join(folder.uri.fsPath, 'app.py'))) {
      return folder.uri.fsPath;
    }
  }
  const repoRoot = path.resolve(extensionPath, '..');
  if (fs.existsSync(path.join(repoRoot, 'app.py'))) {
    return repoRoot;
  }
  return undefined;
}

function pythonCandidates(configured) {
  if (configured) {
    return [configured];
  }
  if (process.platform === 'win32') {
    return ['py', 'python', 'python3'];
  }
  return ['python3', 'python'];
}

function spawnArgs(pythonBin, extra) {
  if (pythonBin === 'py' || pythonBin.endsWith('\\py.exe') || pythonBin.endsWith('/py.exe')) {
    return ['-3', ...extra];
  }
  return extra;
}

function waitForHealth(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(`${url}/health`, (res) => {
        res.resume();
        if (res.statusCode && res.statusCode < 500) {
          resolve();
          return;
        }
        retry();
      });
      req.on('error', retry);
      req.setTimeout(1500, () => {
        req.destroy();
        retry();
      });
    };
    const retry = () => {
      if (Date.now() > deadline) {
        reject(new Error('Server did not become ready. Is rag-debugger installed and is the API key set?'));
        return;
      }
      setTimeout(tick, 400);
    };
    tick();
  });
}

async function getApiKey(context) {
  return context.secrets.get(SECRET_KEY);
}

async function setApiKey(context) {
  const value = await vscode.window.showInputBox({
    title: 'OpenAI API Key',
    prompt: 'Stored in VS Code secret storage. It is not written to the repo.',
    password: true,
    ignoreFocusOut: true,
    placeHolder: 'sk-...',
  });
  if (!value) {
    return false;
  }
  await context.secrets.store(SECRET_KEY, value.trim());
  vscode.window.showInformationMessage('RAG Debugger: API key saved.');
  sidebar?.refresh();
  return true;
}

function stopServer() {
  if (!serverProcess) {
    return;
  }
  const proc = serverProcess;
  serverProcess = undefined;
  if (process.platform === 'win32' && proc.pid) {
    spawn('taskkill', ['/pid', String(proc.pid), '/f', '/t'], { windowsHide: true });
  } else {
    proc.kill();
  }
  if (statusBar) {
    statusBar.text = '$(debug-disconnect) RAG Debugger';
    statusBar.tooltip = 'RAG Debugger stopped';
  }
  sidebar?.refresh();
}

async function startServer(context, options = {}) {
  if (serverProcess) {
    vscode.window.showInformationMessage('RAG Debugger is already running.');
    await openUi();
    return;
  }

  let apiKey = await getApiKey(context);
  if (!apiKey) {
    const setIt = await vscode.window.showWarningMessage(
      'Set your OpenAI API key before starting RAG Debugger.',
      'Set API Key'
    );
    if (setIt === 'Set API Key') {
      const saved = await setApiKey(context);
      if (!saved) {
        return;
      }
      apiKey = await getApiKey(context);
    } else {
      return;
    }
  }

  const { host, port, pythonPath } = getConfig();
  const cwd = findProjectRoot(context.extensionPath);
  const bins = pythonCandidates(pythonPath);
  let lastError = '';

  for (const bin of bins) {
    try {
      serverProcess = spawn(bin, spawnArgs(bin, ['-m', 'rag_debugger', '--host', host, '--port', String(port)]), {
        cwd,
        env: {
          ...process.env,
          OPENAI_API_KEY: apiKey,
          PYTHONUNBUFFERED: '1',
        },
        windowsHide: true,
      });
    } catch (err) {
      lastError = String(err);
      serverProcess = undefined;
      continue;
    }

    let stderr = '';
    serverProcess.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
      if (stderr.length > 4000) {
        stderr = stderr.slice(-4000);
      }
    });
    serverProcess.on('error', (err) => {
      lastError = err.message;
    });
    serverProcess.on('exit', (code) => {
      if (serverProcess) {
        serverProcess = undefined;
        if (code && code !== 0) {
          vscode.window.showErrorMessage(`RAG Debugger exited (${code}). ${stderr.slice(-300)}`);
        }
        sidebar?.refresh();
      }
    });

    try {
      await waitForHealth(serverUrl(), 20000);
      if (statusBar) {
        statusBar.text = '$(pass) RAG Debugger';
        statusBar.tooltip = `Running at ${uiUrl()}`;
      }
      sidebar?.refresh();
      if (!options.skipUi) {
        await openUi();
      }
      return;
    } catch (err) {
      lastError = stderr || err.message;
      stopServer();
    }
  }

  vscode.window.showErrorMessage(
    `Could not start RAG Debugger. Install it with: pip install git+https://github.com/Elyasirankhah/RAG_Debugger.git  (${lastError})`
  );
}

/** @type {object | undefined} */
let lastTrace;

function postJson(pathname, body) {
  const { host, port } = getConfig();
  const payload = JSON.stringify(body);
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: host,
        port,
        path: pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
        },
      },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf8');
          if (res.statusCode && res.statusCode >= 400) {
            reject(new Error(raw || `HTTP ${res.statusCode}`));
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch {
            reject(new Error(raw || 'Invalid JSON from debugger'));
          }
        });
      }
    );
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

function parseTraceFromEditor() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    throw new Error('Open a file and select RAG trace JSON.');
  }
  const raw = editor.document.getText(editor.selection) || editor.document.getText();
  const parsed = JSON.parse(raw);
  if (!parsed.answer || !parsed.retrieved_chunks) {
    throw new Error('Trace JSON needs at least "answer" and "retrieved_chunks".');
  }
  return parsed;
}

function showReport(title, data) {
  const channel = vscode.window.createOutputChannel('RAG Debugger');
  channel.clear();
  channel.appendLine(title);
  channel.appendLine(JSON.stringify(data, null, 2));
  channel.show(true);
}

async function diagnoseFailure(context) {
  if (!serverProcess) {
    await startServer(context, { skipUi: true });
    showReport('Diagnose', report);
    const cause = report.root_cause || 'unknown';
    vscode.window.showInformationMessage(`RAG Debugger root cause: ${cause}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Diagnose failed: ${err.message}`);
  }
}

async function testSuggestedFix(context) {
  if (!serverProcess) {
    await startServer(context, { skipUi: true });
    try {
      lastTrace = parseTraceFromEditor();
    } catch (err) {
      vscode.window.showErrorMessage(`RAG Debugger: ${err.message}`);
      return;
    }
  }
  try {
    const report = await postJson('/repair', lastTrace);
    showReport('Test suggested fix', report);
    const before = report.experiments && report.experiments[0] ? report.experiments[0].supported_pct : '?';
    const after = report.best ? report.best.supported_pct : '?';
    vscode.window.showInformationMessage(`Supported ${before}% → ${after}% (${report.best && report.best.name})`);
  } catch (err) {
    vscode.window.showErrorMessage(`Repair failed: ${err.message}`);
  }
}

class SidebarProvider {
  /**
   * @param {vscode.ExtensionContext} context
   */
  constructor(context) {
    this.context = context;
    this._view = undefined;
  }

  /**
   * @param {vscode.WebviewView} webviewView
   */
  resolveWebviewView(webviewView) {
    this._view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.onDidReceiveMessage(async (message) => {
      switch (message.type) {
        case 'start':
          await startServer(this.context);
          break;
        case 'stop':
          stopServer();
          vscode.window.showInformationMessage('RAG Debugger stopped.');
          break;
        case 'open':
          await openUi();
          break;
        case 'setKey':
          await setApiKey(this.context);
          break;
        case 'diagnose':
          await diagnoseFailure(this.context);
          break;
        case 'testFix':
          await testSuggestedFix(this.context);
          break;
      }
    });
    this.refresh();
  }

  async refresh() {
    if (!this._view) {
      return;
    }
    const running = Boolean(serverProcess);
    const hasKey = Boolean(await getApiKey(this.context));
    this._view.webview.html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
  <style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 12px; }
    h2 { font-size: 14px; margin: 0 0 8px; }
    p { font-size: 12px; opacity: 0.85; line-height: 1.4; }
    button { display: block; width: 100%; margin: 8px 0 0; padding: 8px; cursor: pointer; }
    .ok { color: var(--vscode-testing-iconPassed); }
    .warn { color: var(--vscode-editorWarning-foreground); }
  </style>
</head>
<body>
  <h2>RAG Debugger</h2>
  <p class="${running ? 'ok' : ''}">${running ? 'Server is running.' : 'Server is stopped.'}</p>
  <p class="${hasKey ? 'ok' : 'warn'}">${hasKey ? 'API key is saved in VS Code secret storage.' : 'No API key saved yet.'}</p>
  <button id="key">Set OpenAI API Key</button>
  <button id="start">${running ? 'Restart / Open' : 'Start'}</button>
  <button id="open" ${running ? '' : 'disabled'}>Open demo UI</button>
  <button id="diagnose" ${running ? '' : 'disabled'}>Diagnose RAG failure</button>
  <button id="testFix" ${running ? '' : 'disabled'}>Test suggested fix</button>
  <button id="stop" ${running ? '' : 'disabled'}>Stop</button>
  <p>Select trace JSON in the editor, then Diagnose. Test suggested fix replays larger k / hybrid retrieval.</p>
  <script>
    const vscode = acquireVsCodeApi();
    document.getElementById('key').onclick = () => vscode.postMessage({ type: 'setKey' });
    document.getElementById('start').onclick = () => vscode.postMessage({ type: 'start' });
    document.getElementById('open').onclick = () => vscode.postMessage({ type: 'open' });
    document.getElementById('diagnose').onclick = () => vscode.postMessage({ type: 'diagnose' });
    document.getElementById('testFix').onclick = () => vscode.postMessage({ type: 'testFix' });
    document.getElementById('stop').onclick = () => vscode.postMessage({ type: 'stop' });
  </script>
</body>
</html>`;
  }
}

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  sidebar = new SidebarProvider(context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('ragDebugger.sidebar', sidebar)
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('ragDebugger.setApiKey', () => setApiKey(context))
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('ragDebugger.start', () => startServer(context))
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('ragDebugger.stop', () => {
      stopServer();
      vscode.window.showInformationMessage('RAG Debugger stopped.');
    })
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('ragDebugger.open', () => openUi())
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('ragDebugger.diagnose', () => diagnoseFailure(context))
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('ragDebugger.testFix', () => testSuggestedFix(context))
  );

  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 80);
  statusBar.text = '$(debug-disconnect) RAG Debugger';
  statusBar.command = 'ragDebugger.start';
  statusBar.tooltip = 'Start RAG Debugger';
  statusBar.show();
  context.subscriptions.push(statusBar);
  context.subscriptions.push({ dispose: stopServer });
}

function deactivate() {
  stopServer();
}

module.exports = { activate, deactivate };
