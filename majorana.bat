G:
cd "G:\My Drive\GROWTH\Quantum computing\Majorana Project"
explorer .
start chrome --profile-directory="Profile 3"
CALL conda activate bourne
jupyter notebook --browser "C:/Program Files/Google/Chrome/Application/chrome.exe --profile-directory='Profile 3' %%s"

@REM "'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe' %%s"
@REM """'C:/Program Files/Google/Chrome/Application/chrome.exe --profile-directory="Profile 2"' %%s"""



@REM u'"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" %s'
@REM u'C:/Program Files/Google/Chrome/Application/chrome.exe --profile-directory="Profile 3" %s'