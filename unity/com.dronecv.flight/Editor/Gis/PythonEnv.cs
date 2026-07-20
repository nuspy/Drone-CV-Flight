// Self-contained Python environment for the GIS Environment Builder.
//
// Goal (product requirement): the user needs NOTHING pre-installed. This class
// provisions a private Python + the existing `dronecv` package (with the [gis]
// extra) automatically, so the Editor GUI can drive the SAME Python pipeline
// (`dronecv gis build/export-scene/view`) that the CLI and the PySide6 GUI use
// — no generation/export logic is re-implemented in C#.
//
// Everything lives under <Project>/Library/DroneCV/py (outside Assets/, never
// imported as a Unity asset). Provisioning order:
//   1. a managed venv already present + `dronecv` importable  -> use it
//   2. a system Python >= 3.11 with `venv`                    -> make the venv
//   3. neither                                                -> download a
//      standalone CPython (astral-sh/python-build-standalone), which bundles
//      pip + venv, then make the venv
// then `pip install "dronecv[gis]"` from the bundled/located source.
//
// It never touches the system Python; it is idempotent and resumable.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace DroneCV.Flight.Editor.Gis
{
    public static class PythonEnv
    {
        // --- pinned standalone CPython, used ONLY when no system Python fits.
        // Update these two together (see github.com/astral-sh/python-build-standalone/releases).
        // Overridable via the env var DRONECV_PYTHON_STANDALONE_URL for air-gapped installs.
        const string PbsTag = "20250612";
        const string PyVersion = "3.12.4";

        public static string ManagedRoot =>
            Path.GetFullPath(Path.Combine("Library", "DroneCV", "py"));

        static string VenvDir => Path.Combine(ManagedRoot, "venv");
        static string RuntimeDir => Path.Combine(ManagedRoot, "runtime");

        /// Path to the venv's python executable (may not exist yet).
        public static string VenvPython =>
            Application.platform == RuntimePlatform.WindowsEditor
                ? Path.Combine(VenvDir, "Scripts", "python.exe")
                : Path.Combine(VenvDir, "bin", "python");

        /// True when the managed venv exists and `dronecv` imports.
        public static bool IsReady()
        {
            if (!File.Exists(VenvPython)) return false;
            var (code, _, _) = RunProcess(VenvPython, "-c \"import dronecv\"", null, 60000);
            return code == 0;
        }

        public static string StatusLine()
        {
            if (!File.Exists(VenvPython)) return "not installed";
            var (code, outp, _) = RunProcess(VenvPython,
                "-c \"import dronecv,sys;print(sys.version.split()[0])\"", null, 60000);
            return code == 0 ? $"ready (Python {outp.Trim()})" : "installed, dronecv missing";
        }

        // --------------------------------------------------------- provisioning

        /// Provision the environment (blocking; call from a background thread).
        /// `log` receives progress lines. Throws on unrecoverable failure.
        public static void Ensure(Action<string> log)
        {
            log ??= _ => { };
            Directory.CreateDirectory(ManagedRoot);

            if (IsReady()) { log("Python environment already ready."); return; }

            string basePython = File.Exists(VenvPython) ? VenvPython : ProvisionBasePython(log);

            if (!File.Exists(VenvPython))
            {
                log("Creating virtual environment…");
                var (code, _, err) = RunProcess(basePython, $"-m venv \"{VenvDir}\"", null, 300000, log);
                if (code != 0 || !File.Exists(VenvPython))
                    throw new Exception("venv creation failed: " + err);
            }

            log("Upgrading pip…");
            RunProcess(VenvPython, "-m pip install --upgrade pip", null, 600000, log);

            string source = LocateDroneCvSource(log);
            log($"Installing dronecv[gis] from {source} (this can take a few minutes)…");
            var target = source.EndsWith(".whl") || source.EndsWith(".tar.gz")
                ? $"\"{source}[gis]\""       // wheel/sdist path + extra
                : $"\"{source}[gis]\"";      // repo dir path + extra
            var (ic, _, ierr) = RunProcess(VenvPython, $"-m pip install {target}", null, 1800000, log);
            if (ic != 0) throw new Exception("pip install failed: " + ierr);

            if (!IsReady()) throw new Exception("environment installed but `dronecv` still not importable");
            log("Python environment ready.");
        }

        /// Find a usable base Python: a system one (>=3.11 with venv), else a
        /// downloaded standalone CPython.
        static string ProvisionBasePython(Action<string> log)
        {
            var sys = FindSystemPython(log);
            if (sys != null) return sys;
            log("No suitable system Python found — downloading a private CPython…");
            return DownloadStandalonePython(log);
        }

        static string FindSystemPython(Action<string> log)
        {
            var candidates = new List<string>();
            if (Application.platform == RuntimePlatform.WindowsEditor)
            {
                candidates.Add("py -3.12"); candidates.Add("py -3.11");
                candidates.Add("python"); candidates.Add("python3");
            }
            else
            {
                candidates.Add("python3.12"); candidates.Add("python3.11");
                candidates.Add("python3"); candidates.Add("python");
            }
            foreach (var c in candidates)
            {
                var sp = c.Split(new[] { ' ' }, 2);
                var exe = sp[0]; var pre = sp.Length > 1 ? sp[1] + " " : "";
                var (code, outp, _) = RunProcess(exe,
                    pre + "-c \"import sys,venv;print('%d.%d'%sys.version_info[:2])\"", null, 30000);
                if (code != 0) continue;
                var v = outp.Trim();
                if (VersionAtLeast(v, 3, 11))
                {
                    log($"Using system Python {v} ({c}).");
                    // Return an invokable: for `py -3.12` we must keep the launcher form.
                    return pre.Length > 0 ? exe + " " + pre.Trim() : exe;
                }
            }
            return null;
        }

        static bool VersionAtLeast(string v, int major, int minor)
        {
            var p = v.Split('.');
            return p.Length >= 2 && int.TryParse(p[0], out var a) && int.TryParse(p[1], out var b)
                   && (a > major || (a == major && b >= minor));
        }

        static string DownloadStandalonePython(Action<string> log)
        {
            string url = Environment.GetEnvironmentVariable("DRONECV_PYTHON_STANDALONE_URL")
                         ?? StandaloneUrl();
            Directory.CreateDirectory(RuntimeDir);
            string archive = Path.Combine(RuntimeDir, "python.tar.gz");
            log($"Downloading {url}");
            DownloadFile(url, archive, log);

            log("Extracting CPython…");
            var (code, _, err) = RunProcess("tar", $"-xzf \"{archive}\" -C \"{RuntimeDir}\"", null, 600000, log);
            if (code != 0) throw new Exception("tar extraction failed (need tar on PATH): " + err);
            File.Delete(archive);

            var exe = FindPythonUnder(RuntimeDir);
            if (exe == null) throw new Exception("no python executable found in the standalone archive");
            log($"Standalone CPython at {exe}");
            return exe;
        }

        static string StandaloneUrl()
        {
            string triple;
            bool arm = SystemInfo.processorType.ToLowerInvariant().Contains("arm") ||
                       RuntimeInformationIsArm();
            switch (Application.platform)
            {
                case RuntimePlatform.WindowsEditor:
                    triple = "x86_64-pc-windows-msvc-install_only"; break;
                case RuntimePlatform.OSXEditor:
                    triple = (arm ? "aarch64" : "x86_64") + "-apple-darwin-install_only"; break;
                default: // Linux
                    triple = (arm ? "aarch64" : "x86_64") + "-unknown-linux-gnu-install_only"; break;
            }
            return $"https://github.com/astral-sh/python-build-standalone/releases/download/" +
                   $"{PbsTag}/cpython-{PyVersion}+{PbsTag}-{triple}.tar.gz";
        }

        static bool RuntimeInformationIsArm()
        {
            try { return System.Runtime.InteropServices.RuntimeInformation.OSArchitecture
                         == System.Runtime.InteropServices.Architecture.Arm64; }
            catch { return false; }
        }

        static string FindPythonUnder(string root)
        {
            var name = Application.platform == RuntimePlatform.WindowsEditor ? "python.exe" : "python3";
            foreach (var f in Directory.GetFiles(root, name, SearchOption.AllDirectories))
            {
                // prefer bin/ (unix) or the install root (windows)
                if (f.Contains("bin") || Application.platform == RuntimePlatform.WindowsEditor)
                    return f;
            }
            var any = Directory.GetFiles(root, name, SearchOption.AllDirectories);
            return any.Length > 0 ? any[0] : null;
        }

        /// Where to pip-install dronecv from: an override, a bundled wheel/sdist
        /// under the package's Editor/py~/, or the repo checkout containing this
        /// package (found by walking up to a pyproject.toml with name="dronecv").
        static string LocateDroneCvSource(Action<string> log)
        {
            var env = Environment.GetEnvironmentVariable("DRONECV_PY_SOURCE");
            if (!string.IsNullOrEmpty(env) && (File.Exists(env) || Directory.Exists(env))) return env;

            var pkgDir = PackageEditorDir();
            if (pkgDir != null)
            {
                var bundled = Path.Combine(pkgDir, "py~");
                if (Directory.Exists(bundled))
                {
                    foreach (var pat in new[] { "*.whl", "*.tar.gz" })
                    {
                        var hits = Directory.GetFiles(bundled, pat);
                        if (hits.Length > 0) { log($"Using bundled {Path.GetFileName(hits[0])}"); return hits[0]; }
                    }
                    if (File.Exists(Path.Combine(bundled, "pyproject.toml"))) return bundled;
                }
            }

            var repo = FindRepoRoot(pkgDir ?? Directory.GetCurrentDirectory());
            if (repo != null) { log($"Installing from repo checkout {repo}"); return repo; }

            throw new Exception("could not locate the dronecv Python source. Set DRONECV_PY_SOURCE " +
                                "to the repo folder or a wheel, or bundle one under Editor/py~/.");
        }

        static string PackageEditorDir()
        {
            // This file: .../com.dronecv.flight/Editor/Gis/PythonEnv.cs -> Editor dir.
            var guids = AssetDatabase.FindAssets("PythonEnv t:MonoScript");
            foreach (var g in guids)
            {
                var p = AssetDatabase.GUIDToAssetPath(g);
                if (p.EndsWith("Editor/Gis/PythonEnv.cs"))
                    return Path.GetFullPath(Path.Combine(Path.GetDirectoryName(p), ".."));
            }
            return null;
        }

        static string FindRepoRoot(string start)
        {
            var d = new DirectoryInfo(start);
            for (int i = 0; i < 8 && d != null; i++, d = d.Parent)
            {
                var py = Path.Combine(d.FullName, "pyproject.toml");
                if (File.Exists(py) && File.ReadAllText(py).Contains("name = \"dronecv\"")) return d.FullName;
            }
            return null;
        }

        // ------------------------------------------------------------- run + io

        /// Stable working directory for dronecv runs, so `artifacts/gis/<env>`
        /// and `configs/envs/<env>.yaml` land in the same place for build+export.
        public static string WorkspaceDir
        {
            get { var d = Path.Combine(ManagedRoot, "workspace"); Directory.CreateDirectory(d); return d; }
        }

        /// Run `dronecv <args>` in the managed env, streaming output to onLine.
        /// Returns the process exit code (or -1 if the env isn't ready).
        public static int RunDronecv(string args, Action<string> onLine, string cwd = null)
        {
            if (!File.Exists(VenvPython)) { onLine?.Invoke("Python env not installed."); return -1; }
            var (code, _, _) = RunProcess(VenvPython, "-m dronecv.cli.main " + args, cwd ?? WorkspaceDir, 0, onLine);
            return code;
        }

        static void DownloadFile(string url, string dest, Action<string> log)
        {
            using var client = new HttpClient(new HttpClientHandler { UseProxy = true });
            client.Timeout = TimeSpan.FromMinutes(30);
            using var resp = client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
            resp.EnsureSuccessStatusCode();
            using var src = resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult();
            using var fs = File.Create(dest);
            src.CopyTo(fs);
        }

        /// Run a process. `exeMaybeWithArgs` may embed a launcher prefix (e.g.
        /// "py -3.12"); the first token is the executable. timeoutMs==0 => no
        /// timeout. onLine streams stdout+stderr lines live.
        static (int code, string stdout, string stderr) RunProcess(
            string exeMaybeWithArgs, string args, string cwd, int timeoutMs, Action<string> onLine = null)
        {
            string exe = exeMaybeWithArgs; string prefix = "";
            int sp = exeMaybeWithArgs.IndexOf(' ');
            if (sp > 0) { exe = exeMaybeWithArgs.Substring(0, sp); prefix = exeMaybeWithArgs.Substring(sp + 1) + " "; }

            var psi = new ProcessStartInfo
            {
                FileName = exe,
                Arguments = prefix + args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = cwd ?? ManagedRoot,
            };
            var so = new StringBuilder(); var se = new StringBuilder();
            try
            {
                using var p = new Process { StartInfo = psi };
                p.OutputDataReceived += (_, e) => { if (e.Data != null) { so.AppendLine(e.Data); onLine?.Invoke(e.Data); } };
                p.ErrorDataReceived += (_, e) => { if (e.Data != null) { se.AppendLine(e.Data); onLine?.Invoke(e.Data); } };
                p.Start();
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                if (timeoutMs > 0)
                {
                    if (!p.WaitForExit(timeoutMs)) { try { p.Kill(); } catch { } return (-1, so.ToString(), "timed out"); }
                }
                else p.WaitForExit();
                return (p.ExitCode, so.ToString(), se.ToString());
            }
            catch (Exception ex)
            {
                return (-1, so.ToString(), ex.Message);
            }
        }
    }
}
