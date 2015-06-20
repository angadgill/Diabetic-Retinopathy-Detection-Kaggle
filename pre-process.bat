SET in_dir=data\train-1000
SET out_dir=processed
if not exist %out_dir%\%in_dir% mkdir %out_dir%\%in_dir%

SET size=256x256

for %%i in (%in_dir%\*) do convert -fuzz 10%% -trim +repage -resize %size% -gravity center -background black -extent %size% -equalize %%i %out_dir%\%%i
