# Running the linear regression program
You can run the program just like any other MAMBA program except one difference.
Player 0 needs to read `x_data.txt` and player 1 needs to read `y_data.x`.
The data needs to be loaded into `sfix` which can't be done directly in MAMBA.
So an additional step is needed to convert the data using the `float_to_int.py` script.
In practice, this step can be achieved using the command below from the project root directory.
```
# player 0
cat Programs/linear_regression/x_data.txt | ./Programs/linear_regression/float_to_int.py | ./build/Player.x 0 Programs/linear_regression/
# player 1
cat Programs/linear_regression/y_data.txt | ./Programs/linear_regression/float_to_int.py |  ./build/Player.x 1 Programs/linear_regression/
# player 2
./build/Player.x 2 Programs/linear_regression/
```

