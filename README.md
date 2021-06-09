# MLOps Exercises
Filip Sawicki


### How to run
```bash
python3 ./ex1/main.py train
    --<training args>
    --path (optional, model save directory)
    --name (required, without file extension)
python3 ./ex1/main.py evaluate
    --path (optional, model save directory)
    --name (required, without file extension)
```

### Virtual env
```bash
# create environment "env"
mkvirtualenv env
pip3 install -r requirements.txt

# start/stop environment "env"
workon env
deactivate

# update/install packages (on env)
pip3 install <package>
pip3 freeze > requirements.txt
git add, commit, push
```