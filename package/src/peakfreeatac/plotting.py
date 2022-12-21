import pandas as pd


phases = pd.DataFrame({"phase":["train", "validation"], "color":["#888888", "tomato"]}).set_index("phase")


colors = [
    "#0074D9",
    "#FF4136",
    "#FF851B",
    "#2ECC40",
    "#39CCCC",
    "#85144b"
]