echo 'Running autoflake...'
autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --recursive luna/

echo 'Running isort ...'
isort luna/*.py
