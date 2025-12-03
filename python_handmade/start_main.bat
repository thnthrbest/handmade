@echo off
call "D:\GitHub\handmade\python_handmade\env\Scripts\activate.bat"
python D:\GitHub\handmade\python_handmade\main.py
echo Close in 2 Seconds
TIMEOUT /T 2 /NOBREAK
exit