# probability matrix visualization
分别将alignment matrix 和 coverage 画成heatmap和 bar chart.

## dependency
all the required package is in the `requirements.txt` file. you can install the dependency by
`pip install -r requirements.txt `

## usage
```
python graph.py --base your_base_result_path 
```
The ploted PNG file is in the dir :`current_base_dir/graph`


## Note:
chinese font problem:

1. run `check_font.py` to get the avaiable font on your linux system

2. modify the font part in `graph.py`



