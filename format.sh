path=lunanlp

echo 'Running autoflake ...'
find $path -type f -name "*.py" | xargs autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports 

echo 'Running isort ...'
find $path -type f -name "*.py" | xargs isort

# echo 'Running autopep8 ...'
# find $path -type f -name "*.py" | xargs autopep8 -i

echo 'Running yapf ...'
find $path -type f -name "*.py" | xargs yapf -i