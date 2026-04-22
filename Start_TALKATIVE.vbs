Option Explicit

Dim objFSO, objShell, strPath
Dim strMainPath, strPythonPath, strPythonwPath, strCheckCommand, strRunCommand, exitCode

Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

strPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.CurrentDirectory = strPath

strMainPath = strPath & "\main.py"
strPythonPath = strPath & "\venv\Scripts\python.exe"
strPythonwPath = strPath & "\venv\Scripts\pythonw.exe"

If Not objFSO.FileExists(strMainPath) Then
    MsgBox "TALKATIVE could not start because main.py was not found:" & vbCrLf & strMainPath, vbCritical, "TALKATIVE"
    WScript.Quit 1
End If

If Not objFSO.FileExists(strPythonPath) Or Not objFSO.FileExists(strPythonwPath) Then
    MsgBox "TALKATIVE could not start because the virtual environment is incomplete." & vbCrLf & _
           "Expected to find:" & vbCrLf & strPythonPath & vbCrLf & strPythonwPath, vbCritical, "TALKATIVE"
    WScript.Quit 1
End If

' Validate the script first so syntax errors do not fail silently under pythonw.exe.
strCheckCommand = """" & strPythonPath & """ -m py_compile """ & strMainPath & """"
exitCode = objShell.Run(strCheckCommand, 0, True)

If exitCode <> 0 Then
    MsgBox "TALKATIVE could not start because main.py failed validation." & vbCrLf & _
           "Run this in PowerShell to see the Python error:" & vbCrLf & _
           """" & strPythonPath & """ """ & strMainPath & """", vbCritical, "TALKATIVE"
    WScript.Quit exitCode
End If

' Using pythonw.exe hides the window once validation passes.
strRunCommand = """" & strPythonwPath & """ """ & strMainPath & """"
objShell.Run strRunCommand, 0, False
