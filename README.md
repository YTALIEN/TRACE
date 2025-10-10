# TRACE
The source code of "TRACE: A Generalizable Drift Detector for Streaming Data-Driven Optimization"

## Required packages
The code has been tested running under python 3.10.12, with the following packages installed (along with their dependencies):

- numpy==2.3.2
- pandas==2.3.3
- scipy==1.16.1
- torch==2.0.1+cu118
- torchvision==0.15.2+cu118
- tqdm==4.67.1

## Quick Start
Please run the `main.py`

## Creating Custom Datasets and Train new model
We provide the dataset_generate.py script to offer a standardized way to create your own datasets. You only need to define your specific optimization problem, and the script's uniform data generation function will handle the rest.

Here is an example demonstrating how to add a new problem type called "DBG":
```python
        case "DBG":
        """
        Define the problem instance and varying drift types.
        - Instance range: [1, 6]
        - Drift type range: [1, 7]
        """
        # Define the total number of instances and drift types
        ins_num, drf_num = 6, 7

        # Pre-allocate a NumPy array to store the generated data
        # Dimensions: [instance, drift_type, environment, sample, decision_vars + objective_val]
        origin_data = np.empty([ins_num, drf_num, self.max_env, self.n_samples, self.dim + 1])
        
        # Define the lower and upper bounds for the decision variables
        lb, ub = -5, 5

        # Loop through each instance and drift type
        for ins in range(1, ins_num + 1):
            for drf in range(1, drf_num + 1):
                
                # Set the parameters for the optimization problem
                params = {
                    "fun_num": ins,
                    "change_instance": drf,
                    "lb": lb,
                    "ub": ub,
                }
                
                # Generate data for each environment (snapshot in time)
                for env in range(self.max_env):
                    # Generate sample points using Latin Hypercube Sampling (LHS)
                    x = lhs(self.n_samples, self.dim, lb, ub)
                    
                    # Update parameters with current samples and environment count
                    params.update(x=x, change_count=env)
                    
                    # === ACTION REQUIRED ===
                    # Call your custom optimization problem function here.
                    # It should take 'params' and return the objective values.
                    y, _ = DBG(params) 
                    
                    # Store the results in the pre-allocated array
                    origin_data[ins-1, drf-1, env, :, :-1] = x  # Store decision variables
                    origin_data[ins-1, drf-1, env, :, -1] = y   # Store corresponding objective value

```

Then, a new model can be trained by `train_TRACE.py`, replace the `data_path` to your new data path.


