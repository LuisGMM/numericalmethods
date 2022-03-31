
import pandas as pd

#TODO: Account doc | & functionality for the kwargs

def to_latex(col_names: 'list[str, ...]', cols: 'list[list, ...]', **kwargs) -> str:
    """High-level wrapper of pandas.DataFrame.to_latex with limited options.
    Zips the strings in col_names with the sublists in cols and generates a LaTeX table in string format 
    Args:
        col_names (list[str]): The names of each data column in cols. They will appear in the first row.
        cols (list[list]): Contains lists, each a supposed column of the table
    Returns:
        str: Latex table.
    """
    return pd.DataFrame(dict(zip(col_names, cols))).to_latex(**kwargs)


if __name__ == '__main__':

    names = ['luis', 'pablo']
    cols = [list(range(10)), list(range(10))]

    print(to_latex(names, cols))