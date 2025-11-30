@echo off
call "E:\hand-shadow\server\env\Scripts\activate.bat"
python E:\hand-shadow\server\calibrate.py
echo Close in 2 Seconds
TIMEOUT /T 2 /NOBREAK
exit