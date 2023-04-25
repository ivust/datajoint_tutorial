# DataJoint Tutorial

- Clone the repository.
- Install the package using `poetry`:
```
$ poetry install
```
- Start the database Docker container:
```
$ docker run -p 3306:3306 --rm -e MYSQL_ROOT_PASSWORD=password -d mysql:latest
```
- Start Jupyter :
```
$ poetry run jupyter lab
```
- Open the tutorial notebook in `notebooks` and everything should hopefully work.
- For the parallel computation example, the training job can be started as
```
$ poetry run python run_training_job.py
```