using System;
using System.Diagnostics;

string imgPath = args.Length > 0 ? args[0] : "../test.png";

var psi = new ProcessStartInfo
{
    FileName               = "./ppocr/main",
    Arguments              = $"config.json {imgPath}",
    WorkingDirectory       = "./ppocr",       // 让 config.json 里 ./models/ 生效
    RedirectStandardOutput = true,
    RedirectStandardError  = true,
    UseShellExecute        = false,
    CreateNoWindow         = true,
};

using var proc = Process.Start(psi)!;
string stdout = proc.StandardOutput.ReadToEnd();
proc.WaitForExit();

bool found = false;
foreach (var line in stdout.Split('\n'))
{
    var t = line.Trim();
    if (t.Contains("[Text]")) { found = true; continue; }
    if (found && t.Length > 0)
    {
        while (t.StartsWith("[")) { int e = t.IndexOf("] "); if (e < 0) break; t = t.Substring(e + 2); }
        Console.WriteLine(t);
    }
}
