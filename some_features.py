import pandas as pd
import pandas.api.types as ptypes
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, accuracy_score, r2_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_curve, auc)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as plx
import numpy as np
import datetime as dtm
import time


def column_single_view(column_name_list, data=None):
    """we enter the column name that we want to observe separately"""
    if data is not None:
        new_data = data[[col_name for col_name in column_name_list]]

        st.write(new_data)
        with st.expander("See General data_info"):
            st.write(data_info(new_data))
        return new_data


def data_info(data):
    if data is not None:
        lines = len(data)
        cols = len(data.columns)
        col_type = {data.columns[i]: [data[data.columns[i]].dtype] for i in range(len(data.columns))}
        df = pd.DataFrame(col_type, index=["col_type"])
        st.write(f"""This peace of data has:""")
        st.write(f""">{lines} lines;""")
        st.write(f""">{cols} columns;""")
        st.write(f"""See below each columns_type:""")
        st.write(df)
    else:
        st.write("No data found for any info!")


class FileManagement:

    def __init__(self, data_file):
        self.data = data_file
        self.formatted_dt = None
        self.subtitle = ""
        self.memory_diction = {}

    def get_data(self):
        return self.data

    def open_file(self, n=None, drop_null=False, reset_idx=False):
        """Opens the data file based on its extension and returns the DataFrame.
        Returns:
            pandas.DataFrame: The loaded DataFrame or None if an error occurs."""
        if self.data is not None:
            try:
                # Gestion du drop_null
                if drop_null:
                    dt = self.data.dropna() if n is None else self.data.head(n).dropna()
                else:
                    dt = self.data if n is None else self.data.head(n)

                # Gestion du reset_index
                if reset_idx:
                    dt = dt.reset_index(drop=True)

                st.write(dt)
                with st.expander("See General data_info"):
                    st.write(data_info(dt))
                return dt

            except Exception as e:
                return st.write(f"""Sorry!
                                    Something went wrong...
                                    Error:{e}""")
        else:
            st.write("""No file has been found !
                        Kindly upload first...""")
    # __________________________________________________________________________________________________________________

    def transform(self, into, file_name):
        """Transforms the data to the specified format and potentially saves it.
                Args:
                    into (str): The target format ("csv", "excel", "sql", "dictionary").
                    file_name (str): The desired filename for the transformed data.
                Returns:
                    object: The transformed data or None if an error occurs."""

        if self.data is not None:
            try:
                if into == "csv":
                    self.data.to_csv(f"C:\\Users\\nonso\\Desktop\\Coding_projects\\{file_name}.csv", index=False)
                    return st.success("Successful conversion into csv_format !")

                elif into == "excel":
                    self.data.to_excel(f"C:\\Users\\nonso\\Desktop\\Coding_projects\\{file_name}.xlsx", index=False)
                    return st.success("Successful conversion into excel_format !")

                elif into == "sql":
                    # Create a database engine
                    engine = create_engine(f'sqlite:///{file_name}.db')  # Replace with your database connection string

                    ''''sqlite:///my_database.db': Cette chaîne de connexion indique à SQLAlchemy de se connecter 
                    à une base de données SQLite nommée my_database.db qui sera créée dans le répertoire courant 
                    si elle n'existe pas.'''

                    self.data.to_sql(f'{file_name}', engine, index=False)  # Specify the table name and engine
                    return st.success("Successful conversion into sql_format !")

                elif into == "dictionary":
                    self.data.to_dict("records")
                    return st.success("Successful conversion into dictionary_format !")
                else:
                    st.write(f"Unsupported transformation format: {into}")

            except Exception as e:
                return st.write(f"Error during transformation: {e}")

        else:
            return st.write("No data loaded for conversion.")
    # __________________________________________________________________________________________________________________

    def time_filer(self, trgtdt):
        """The time_filer function is designed to filter data based on date and time criteria.
        It is part of a class (indicated by the self parameter) and takes one parameter, trgtdt,
        which is a DataFrame containing the data to be filtered. Here’s a breakdown of its functionality:

        User Interaction:
        >Select the target column from the dropdown menu.
        >Choose the comparison sign (e.g., before, after).
        >Select the filtering mode (Entire period or one_value).
        >Define the required date or time values based on the selected mode.

        Get Filtered Data:
        >The function will return the filtered DataFrame based on the criteria provided."""

        if trgtdt is not None:

            dtcol = [x for x in trgtdt if pd.api.types.is_datetime64_any_dtype(trgtdt[x]) is True]

            if len(dtcol) < 1:
                st.write("""No column has a date_time data_type. If you see any column with a date_format,
                you need to formate it first (into date/time)""")

            else:
                fin_val = None
                valuex = None
                seq = {"year": None, "month": None, "day": None, "hour": None, "minute": None, "second": None}
                main_col = st.selectbox("Pick the column of  your target", [None]+dtcol)
                signe = st.selectbox("I want date that are:", [None, "before(<)", "before or on(<=)", "on(==)",
                                                               "after(>)", "after or on(>=)", "different(!=)"])
                mode = st.selectbox("Chose the filtering mode:", [None, "Entire period", "one_value"])

                if mode == "Entire period":
                    valuex = "Entire period"
                    for x in seq.keys():
                        if x == "year":
                            val = st.number_input(f"Define the {x}", min_value=0)
                            seq[x] = val
                        elif x == "month":
                            val = st.number_input(f"Define the {x}", min_value=1, max_value=12)
                            seq[x] = val
                        elif x == "day":
                            val = st.number_input(f"Define the {x}", min_value=1, max_value=31)
                            seq[x] = val
                        elif x == "hour":
                            val = st.number_input(f"Define the {x}", min_value=0, max_value=23)
                            seq[x] = val
                        elif x == "minute":
                            val = st.number_input(f"Define the {x}", min_value=0, max_value=59)
                            seq[x] = val
                        elif x == "second":
                            val = st.number_input(f"Define the {x}", min_value=0, max_value=59)
                            seq[x] = val

                    fin_val = dtm.datetime(year=seq["year"], month=seq["month"], day=seq["day"],
                                           hour=seq["hour"], minute=seq["minute"], second=seq["second"])
                    fin_val = pd.Timestamp(fin_val)
                    st.success(f"Filtering value: **{fin_val}**")

                elif mode == "one_value":
                    val = None
                    tmseq = st.selectbox("Pick the time_value of your target", [None]+[x for x in seq.keys()])
                    valuex = tmseq
                    if tmseq == "year":
                        val = st.number_input(f"Define the {tmseq}", min_value=00)
                    elif tmseq == "month":
                        val = st.number_input(f"Define the {tmseq}", min_value=1, max_value=12)
                    elif tmseq == "day":
                        val = st.number_input(f"Define the {tmseq}", min_value=1, max_value=31)
                    elif tmseq == "hour":
                        val = st.number_input(f"Define the {tmseq}", min_value=0, max_value=23)
                    elif tmseq == "minute":
                        val = st.number_input(f"Define the {tmseq}", min_value=00, max_value=59)
                    elif tmseq == "second":
                        val = st.number_input(f"Define the {tmseq}", min_value=00, max_value=59)

                    fin_val = val

                if fin_val is not None:
                    return FileManagement.time_comp(self, data=trgtdt, column_ref=main_col,
                                                    comparator=signe, pointed_val=fin_val, valuex=valuex)
    # __________________________________________________________________________________________________________________

    def time_comp(self, data, column_ref, comparator, pointed_val, valuex):
        """The time_comp function filters data based on specific date and time criteria. It takes five parameters:

        -data: The DataFrame containing the data to be filtered.
        -column_ref: The column in the DataFrame that contains the date/time information.
        -comparator: The comparison operation (e.g., "before", "after", "on") to apply.
        -pointed_val: The specific date/time value to compare against.
        -valuex: Specifies if the filtering is for an entire period or a single time component (e.g., year, month).

        >The function will return and display the filtered DataFrame based on the criteria provided."""
        dt = None
        try:

            if comparator == "on(==)":
                if valuex == "Entire period":
                    dt = data[data[column_ref] == pointed_val]
                elif valuex == "year":
                    dt = data[data[column_ref].dt.year == pointed_val]
                elif valuex == "month":
                    dt = data[data[column_ref].dt.month == pointed_val]
                elif valuex == "day":
                    dt = data[data[column_ref].dt.day == pointed_val]
                elif valuex == "hour":
                    dt = data[data[column_ref].dt.hour == pointed_val]
                elif valuex == "minute":
                    dt = data[data[column_ref].dt.minute == pointed_val]
                elif valuex == "second":
                    dt = data[data[column_ref].dt.second == pointed_val]

            elif comparator == "different(!=)":
                if valuex == "Entire period":
                    dt = data[data[column_ref] != pointed_val]
                elif valuex == "year":
                    dt = data[data[column_ref].dt.year != pointed_val]
                elif valuex == "month":
                    dt = data[data[column_ref].dt.month != pointed_val]
                elif valuex == "day":
                    dt = data[data[column_ref].dt.day != pointed_val]
                elif valuex == "hour":
                    dt = data[data[column_ref].dt.hour != pointed_val]
                elif valuex == "minute":
                    dt = data[data[column_ref].dt.minute != pointed_val]
                elif valuex == "second":
                    dt = data[data[column_ref].dt.second != pointed_val]

            elif comparator == "before(<)":
                if valuex == "Entire period":
                    dt = data[data[column_ref] < pointed_val]
                elif valuex == "year":
                    dt = data[data[column_ref].dt.year < pointed_val]
                elif valuex == "month":
                    dt = data[data[column_ref].dt.month < pointed_val]
                elif valuex == "day":
                    dt = data[data[column_ref].dt.day < pointed_val]
                elif valuex == "hour":
                    dt = data[data[column_ref].dt.hour < pointed_val]
                elif valuex == "minute":
                    dt = data[data[column_ref].dt.minute < pointed_val]
                elif valuex == "second":
                    dt = data[data[column_ref].dt.second < pointed_val]

            elif comparator == "before or on(<=)":
                if valuex == "Entire period":
                    dt = data[data[column_ref] <= pointed_val]
                elif valuex == "year":
                    dt = data[data[column_ref].dt.year <= pointed_val]
                elif valuex == "month":
                    dt = data[data[column_ref].dt.month <= pointed_val]
                elif valuex == "day":
                    dt = data[data[column_ref].dt.day <= pointed_val]
                elif valuex == "hour":
                    dt = data[data[column_ref].dt.hour <= pointed_val]
                elif valuex == "minute":
                    dt = data[data[column_ref].dt.minute <= pointed_val]
                elif valuex == "second":
                    dt = data[data[column_ref].dt.second <= pointed_val]

            elif comparator == "after(>)":
                if valuex == "Entire period":
                    dt = data[data[column_ref] > pointed_val]
                elif valuex == "year":
                    dt = data[data[column_ref].dt.year > pointed_val]
                elif valuex == "month":
                    dt = data[data[column_ref].dt.month > pointed_val]
                elif valuex == "day":
                    dt = data[data[column_ref].dt.day > pointed_val]
                elif valuex == "hour":
                    dt = data[data[column_ref].dt.hour > pointed_val]
                elif valuex == "minute":
                    dt = data[data[column_ref].dt.minute > pointed_val]
                elif valuex == "second":
                    dt = data[data[column_ref].dt.second > pointed_val]

            elif comparator == "after or on(>=)":
                if valuex == "Entire period":
                    dt = data[data[column_ref] >= pointed_val]
                elif valuex == "year":
                    dt = data[data[column_ref].dt.year >= pointed_val]
                elif valuex == "month":
                    dt = data[data[column_ref].dt.month >= pointed_val]
                elif valuex == "day":
                    dt = data[data[column_ref].dt.day >= pointed_val]
                elif valuex == "hour":
                    dt = data[data[column_ref].dt.hour >= pointed_val]
                elif valuex == "minute":
                    dt = data[data[column_ref].dt.minute >= pointed_val]
                elif valuex == "second":
                    dt = data[data[column_ref].dt.second >= pointed_val]

            st.write(dt)
            with st.expander("See the General data_info"):
                st.write(data_info(dt))
            return dt

        except Exception as e:
            return st.write(f'Something went bad: {e}')
    # __________________________________________________________________________________________________________________

    def filtering(self, column_ref, comparator, pointed_val):
        """We should respect the three parameter when applying the filtering method:
        column_ref: placed in first, designs the column containing the targeted value (the value that we want to observe
        separately, that value is also a parameter called the pointed_val, placed at the third place.)
        comparator: placed in second, should always be expressed in  signs(==, !=, >=, <=) depending on if the
        pointed_val is a string or a numerical value (int or float)"""

        try:
            if column_ref in self.data.columns:
                dt = None
                try:
                    if pointed_val is not None:
                        if comparator == "==":
                            dt = self.data[self.data[column_ref].isin(pointed_val)]

                        elif comparator == "!=":
                            dt = self.data[~self.data[column_ref].isin(pointed_val)]

                        elif comparator == ">":
                            if all((isinstance(i, int) or isinstance(i, float)) for i in list(self.data[column_ref])):
                                dt = self.data[self.data[column_ref] > pointed_val[0]]

                            else:
                                st.write("Some string character detected, no way to compare using '>'")
                        elif comparator == ">=":
                            if all((isinstance(i, int) or isinstance(i, float)) for i in list(self.data[column_ref])):
                                dt = self.data[self.data[column_ref] >= pointed_val[0]]

                            else:
                                st.write("Some string character detected, no way to compare using '>='")
                        elif comparator == "<":
                            if all((isinstance(i, int) or isinstance(i, float)) for i in list(self.data[column_ref])):
                                dt = self.data[self.data[column_ref] < pointed_val[0]]

                            else:
                                st.write("Some string character detected, no way to compare using '<'")
                        elif comparator == "<=":
                            if all((isinstance(i, int) or isinstance(i, float)) for i in list(self.data[column_ref])):
                                dt = self.data[self.data[column_ref] <= pointed_val[0]]

                        else:
                            st.write("Some string character detected, no way to compare using '<='")

                    st.write(dt)
                    with st.expander("See data_info"):
                        st.write(data_info(dt))
                    return dt

                except Exception as e:
                    return st.write(f'Something went bad: {e}')

            else:
                return st.write("The column's name provided is not found, kindly check...")

        except Exception as e:
            return st.write(f"Something went bad: {e}")
    # __________________________________________________________________________________________________________________

    def line_single_view(self, by_index, to_index=None):
        """by_index: the index of the line we want to observe if to_index is kept None, if to_index is defined, by_index
         designs the starting line index and to_index designs the ending line index """

        if to_index is None:
            rep = self.data.loc[by_index]

        elif to_index == 'max':
            rep = self.data.loc[by_index:]

        elif to_index >= by_index:
            rep = self.data.loc[by_index:to_index]

        else:
            rep = """The 'to_index-value' should be either None, either Max, or an integer greater or equal to the 
                     'by_index'"""

        st.write(rep)
        with st.expander("See General data_info"):
            st.write(data_info(rep))
        return rep
    # __________________________________________________________________________________________________________________

    def data_formatting(self, trgtdt):
        st.write(trgtdt)
        if trgtdt is not None and not trgtdt.empty:
            try:

                form = st.selectbox("Pick a formatting action", [None, "Data_type formatting", "Data_updating"])

                if form == "Data_type formatting":

                    col = st.multiselect("Pick the column to be formatted", [x for x in trgtdt.columns])
                    dtyp = st.selectbox("The selected column are to format into:",
                                        ["None", "integer", "float", "object(str, category)", "date/time(entire)"])
                    if dtyp == "integer":
                        for slc in col:
                            trgtdt[slc] = trgtdt[slc].astype(int)
                        st.success("SUCCESSFUL CONVERSION !")

                    elif dtyp == "float":
                        for slc in col:
                            trgtdt[slc] = trgtdt[slc].astype(float)
                        st.success("SUCCESSFUL CONVERSION !")

                    elif dtyp == "object(str, category)":
                        for slc in col:
                            trgtdt[slc] = trgtdt[slc].astype(object)
                        st.success("SUCCESSFUL CONVERSION !")

                    elif dtyp == "date/time(entire)":
                        for slc in col:
                            trgtdt[slc] = pd.to_datetime(trgtdt[slc])
                        st.success("SUCCESSFUL CONVERSION !")

                elif form == "Data_updating":
                    def resolve(a, b, ope):
                        rep = None
                        if ope == "add":
                            rep = a + b
                        elif ope == "subtract":
                            rep = a - b
                        elif ope == "multiply":
                            rep = a * b
                        elif ope == "divide":
                            rep = a / b
                        elif ope == "exponent":
                            rep = a ** b

                        return rep

                    action = st.selectbox("What do you need to do?", [None, "Modification", "Create new column"])

                    if action == "Modification":
                        to_do = st.selectbox("Select the to_do", [None, "Modify column's values", "Modify column's name"])

                        if to_do == "Modify column's values":
                            col = st.multiselect("Which column(s) do you need to modify?", [x for x in trgtdt.columns])
                            for xco in col:
                                st.write(f"The *{xco}* column")

                                if trgtdt[xco].dtype == object:
                                    st.text("""This is an object_data_type column.
                                    Everything written will be taken as a string and will be added or retrieved  
                                    to each element of this column.""")

                                    text = st.text_input("Enter the value to be added here")
                                    bts = st.radio("", ["add", "delete"])
                                    if bts == "add":
                                        trgtdt[xco] = trgtdt[xco] + text
                                        st.success('Done!')
                                        st.write(trgtdt[xco])

                                    elif bts == "delete":
                                        trgtdt[xco] = trgtdt[xco].str.replace(text, '', regex=False)
                                        st.success('Done!')
                                        st.write(trgtdt[xco])

                                elif trgtdt[xco].dtype == int or trgtdt[xco].dtype == float:
                                    st.text("""This is a numeric_data_type column.
                                    Only mathematical operations can be performed.
                                    The selected operation will be applied to each element of this column.""")

                                    scrcol1, scrcol2 = st.columns([1, 5])

                                    with scrcol1:
                                        op = st.radio("Pick an operation",
                                                      [None, "add", "subtract", "multiply", "divide", "exponent"])

                                    with scrcol2:
                                        term2 = st.selectbox("Define the 2nd term value",
                                                             ["Specify by yourself", "Apply to another column"])

                                        if term2 == "Specify by yourself":
                                            term1 = st.number_input("Enter the value")

                                            if st.checkbox("apply"):
                                                trgtdt[xco] = resolve(trgtdt[xco], term1, op)
                                                st.success('Done!')
                                                st.write(trgtdt[xco])

                                        elif term2 == "Apply to another column":
                                            term = (st.selectbox("To which column?",
                                                [None]+[x for x in trgtdt.columns if ptypes.is_numeric_dtype(trgtdt[x])]))

                                            if st.checkbox("apply"):
                                                trgtdt[xco] = resolve(trgtdt[xco], trgtdt[term], op)
                                                st.success('Done!')
                                                st.write(trgtdt[xco])

                                elif pd.api.types.is_datetime64_any_dtype(trgtdt[xco]) is True:
                                    st.text("""This is a date_time data_type column...
                                    The only thing you can perform is to modify de time format, which will be applied
                                    to each of the element of this column""")

                                    period = st.selectbox("time period_section", ["day-month-year", "hour-min-sec"])

                                    if period == "day-month-year":
                                        dtform = st.radio("pick a style", [None, "yy-mm-dd", "dd-mm-yy"])

                                        if dtform == "yy-mm-dd":
                                            for slc in col:
                                                trgtdt[slc] = pd.to_datetime(trgtdt[slc], format='%Y-%m-%d',
                                                                             errors='coerce')
                                            st.success("SUCCESSFUL CONVERSION !")

                                        elif dtform == "dd-mm-yy":
                                            for slc in col:
                                                trgtdt[slc] = pd.to_datetime(trgtdt[slc], format='%d-%m-%Y',
                                                                             errors='coerce')
                                            st.success("SUCCESSFUL CONVERSION !")

                                    elif period == "hour-min-sec":
                                        for slc in col:
                                            trgtdt[slc] = pd.to_datetime(trgtdt[slc], format='%H:%M:%S',
                                                                         errors='coerce').dt.time
                                        st.success("SUCCESSFUL CONVERSION !")

                        elif to_do == "Modify column's name":
                            col = st.multiselect("Which column(s) do you need to modify?", [x for x in trgtdt.columns])
                            name_dico = {}

                            for old_name in col:
                                st.write(f"Modifying {old_name}")
                                new_name = st.text_input("Enter the new name here", key=f"{old_name}")
                                name_dico[old_name] = new_name

                            proceed = st.radio("Proceed", ["Verify", "Apply"])
                            if proceed == "Apply" and len(name_dico) > 0:
                                trgtdt = trgtdt.rename(columns=name_dico)
                                st.success("Successfully Done!")
                            else:
                                st.write("Nothing has been changed yet...")

                    elif action == "Create new column":
                        col_name = st.text_input("Enter the column_name")
                        col = st.selectbox("From which column will you go?", [None]+[x for x in trgtdt.columns])

                        if trgtdt[col].dtype == object:
                            st.text("""This is an object_data_type column.
                                    Everything written will be taken as a string and will be added or retrieved  
                                    to each element of this column; then the result will build the new column""")

                            text = st.text_input("Enter the value to be added/deleted here")
                            bts = st.radio("", [None, "add", "delete"])

                            if bts == "add":
                                trgtdt[col_name] = trgtdt[col] + text
                                st.success('Done!')
                                st.write(trgtdt[col_name])

                            elif bts == "delete":
                                trgtdt[col_name] = trgtdt[col].str.replace(text, '', regex=False)
                                st.success('Done!')
                                st.write(trgtdt[col_name])

                        elif ptypes.is_numeric_dtype(trgtdt[col]):
                            st.text("""This is a numeric_data_type column.
                                    Only mathematical operations can be performed.
                                    The selected operation will be applied to each element of this column;
                                    then the result will build the new column""")

                            scrcol1, scrcol2 = st.columns([1, 5])

                            with scrcol1:
                                op = st.radio("Pick an operation",
                                              [None, "add", "subtract", "multiply", "divide", "exponent"])

                            with scrcol2:
                                term = st.selectbox("Define the 2nd term value",
                                                    ["Specify by yourself", "Apply to another column"])

                                if term == "Specify by yourself":
                                    term1 = st.number_input("Enter the value")
                                    if st.checkbox("apply"):
                                        trgtdt[col_name] = resolve(trgtdt[col], term1, op)
                                        st.success('Done!')
                                        st.write(trgtdt[col_name])

                                elif term == "Apply to another column":
                                    term2 = st.selectbox("To which column?", [None]+[x for x in trgtdt.columns if
                                                         ptypes.is_numeric_dtype(trgtdt[x])])
                                    if st.checkbox("apply"):
                                        trgtdt[col_name] = resolve(trgtdt[col], trgtdt[term2], op)
                                        st.success('Done!')
                                        st.write(trgtdt[col_name])

                        elif pd.api.types.is_datetime64_any_dtype(trgtdt[col]) is True:
                            st.text("""This is a date_time data_type column...
                            The only thing you can perform is to create separate values_columns such as year's column, 
                            hour's column, etc ie. to extract specific values""")

                            tmval = (st.selectbox("Which time_values do need to extract?",
                                     [None, "year", "month", "day", "hour", "minute", "second", "days_name",
                                      "months_name", "date only", "time only"]))

                            if st.checkbox("apply"):
                                if tmval is None:
                                    st.write("You've not chose any potion yet!")
                                elif tmval == "year":
                                    trgtdt[col_name] = trgtdt[col].dt.year
                                elif tmval == "month":
                                    trgtdt[col_name] = trgtdt[col].dt.month
                                elif tmval == "day":
                                    trgtdt[col_name] = trgtdt[col].dt.day
                                elif tmval == "hour":
                                    trgtdt[col_name] = trgtdt[col].dt.hour
                                elif tmval == "minute":
                                    trgtdt[col_name] = trgtdt[col].dt.minute
                                elif tmval == "second":
                                    trgtdt[col_name] = trgtdt[col].dt.second
                                elif tmval == "days_name":
                                    trgtdt[col_name] = trgtdt[col].dt.day_name()
                                elif tmval == "months_name":
                                    trgtdt[col_name] = trgtdt[col].dt.month_name()
                                elif tmval == "date only":
                                    trgtdt[col_name] = trgtdt[col].dt.date
                                elif tmval == "time only":
                                    trgtdt[col_name] = trgtdt[col].dt.time

                                st.success('Done!')
                                st.write(trgtdt[col_name])

                with st.expander("See General data_info"):
                    st.write(data_info(trgtdt))

                if st.button("Submit formatted data"):
                    self.data = trgtdt
                    st.success("Successful submission!")

                proc = st.radio("Keep the formatted data as a new file", [None, "proceed"])

                if proc == "proceed":
                    fil_n = st.text_input("Enter the new file's name")
                    fil_tp = st.radio("Keep as:", [None, "csv", "excel"])
                    if st.checkbox("create"):
                        if fil_n is not None and fil_tp is not None:
                            FileManagement.transform(self, into=fil_tp, file_name=fil_n)
                            st.success(f"A new file {fil_n}.{fil_tp} has been created !")
                        else:
                            st.write("One of the required field is not filled !")

            except Exception as e:
                st.write(f"{e}")
        else:
            st.write("No data found yet...")
    # __________________________________________________________________________________________________________________

    def data_clearing(self, trgtdt):
        with st.expander("The logic behind"):
            st.write(
                """The data_clearing function is designed to clean and filter data based on user-defined criteria.
                It is part of a class (indicated by the self parameter) and takes one parameter, trgtdt,
                which is a DataFrame containing the data to be processed. Here’s a breakdown of its functionality:
        
                >>Select Delimitation Method:
                --The user selects a delimitation method from a dropdown menu. The options are:
        
                -The entire data: Process the entire DataFrame.
                -n_first row(s): Process the first n rows of the DataFrame.
                -Filter: Apply specific filters to the data.
        
                >>Process Entire Data:
                If the user selects "The entire data," the function calls FileManagement.open_file to open and process
                the entire DataFrame.
        
                >>Process First n Rows:
                If the user selects "n_first row(s)," they are prompted to enter the number of rows to process.
                The function then calls FileManagement.open_file with the specified number of rows.
        
                >>Apply Filters:
                If the user selects "Filter," they can choose from three filtering modes:
        
                -Row/Column filtering: Allows the user to filter specific rows and columns.
                -Data_value filtering: Filters data based on specific values in a column.
                -Time_filtering: Filters data based on date and time criteria.
        
                >>Return Filtered Data:
                The function returns the filtered DataFrame based on the selected criteria.""")

        df = None

        delimitation = st.selectbox("Select a delimitation method",
                                    ["The entire data", "n_first row(s)", "Filter"])

        part1, part2 = st.columns([5, 5])

        with part1:
            dropping_null = st.radio("Drop Null(NaN):", ["No", "yes"], key="drop_null")
        with part2:
            reset_index = st.radio("Reset the index:", ["No", "yes"], key="reset_index")

        if trgtdt is not None and not trgtdt.empty:

            if delimitation == "The entire data":
                if dropping_null == "yes":
                    st.success("Note that lines containing null values have been dropped")
                    df = FileManagement.open_file(self, drop_null=True, reset_idx=(reset_index == "yes"))
                else:
                    df = FileManagement.open_file(self, reset_idx=(reset_index == "yes"))
                self.subtitle = "The entire data"

            elif delimitation == "n_first row(s)":
                n = st.number_input("Enter the limitation row", min_value=1, max_value=len(trgtdt))
                if dropping_null == "yes":
                    st.success("Note that lines containing null values have been dropped")
                    df = FileManagement.open_file(self, n, drop_null=True, reset_idx=(reset_index == "yes"))
                else:
                    df = FileManagement.open_file(self, n, reset_idx=(reset_index == "yes"))
                self.subtitle = f"The {n}_first row(s)"

            elif delimitation == "Filter":
                filtmod = st.selectbox("Select a filtering mode",
                                       ["Row/Column filtering", "Data_value filtering", "Time_filtering"])

                if filtmod == "Row/Column filtering":
                    prime = None

                    '''>>>>Row setting'''
                    try:
                        starting_line = st.number_input("Enter the starting line rank", min_value=0,
                                                        max_value=len(trgtdt) - 1)
                        next_step = st.radio("", ["Display the line", "Define the ending line rank"])

                        if next_step == "Display the line":
                            prime = FileManagement.line_single_view(self, starting_line)
                        elif next_step == "Define the ending line rank":
                            next_step2 = st.radio("", ["Go to Max", "Go through"])

                            if next_step2 == "Go to Max":
                                prime = FileManagement.line_single_view(self, starting_line, "max")
                            elif next_step2 == "Go through":
                                ending_line = st.number_input("Enter the ending line rank", min_value=starting_line,
                                                              max_value=len(trgtdt) - 1)
                                prime = FileManagement.line_single_view(self, starting_line, ending_line)

                    except Exception as e:
                        st.write(f"{e}")

                    '''>>>>Column setting'''
                    try:
                        if prime is not None:

                            target_col = st.multiselect("Select all the columns of your need",
                                                        [col for col in pd.DataFrame(prime).columns])

                            df = column_single_view(target_col, prime)

                    except Exception as e:
                        st.write(f"{e}")

                elif filtmod == "Data_value filtering":
                    try:
                        main_col = st.selectbox("The main column to filter is:",
                                                [None]+[col for col in trgtdt.columns])
                        main_comp = st.selectbox(f"In this '{main_col}' column, check all values that are:",
                                                 [None]+["==", "!=", ">", ">=", "<", "<="])
                        main_val = st.multiselect(f"""In this '{main_col}' column, check all values that are {main_comp}
                                                to/from/than:""", [val for val in trgtdt[main_col].unique()])
                    except Exception as e:
                        st.write(f"We're facing this error: {e}. Try to solve it...")
                    else:
                        if main_col is not None and main_comp is not None and main_val is not None:
                            df = FileManagement.filtering(self, main_col, main_comp, main_val)
                            self.subtitle = f"""Data filtered in the '{main_col}' column, for values {main_comp} 
                            to/from/than *{main_val}*"""

                elif filtmod == "Time_filtering":
                    try:
                        df = FileManagement.time_filer(self, trgtdt)
                    except Exception as e:
                        st.write({e})

                if df is not None:
                    if st.button("Use this as main data"):
                        self.data = df
                        st.success("""
                                    Successful submission!
                                    Go back to the section <Entire data> to see the updating """)
            return df

        else:
            st.write("No file uploaded yet")
    # __________________________________________________________________________________________________________________

    def quartile_percentile(self, main_data):
        try:
            if isinstance(main_data, pd.DataFrame) and main_data is not None and not main_data.empty:
                observatory_col = st.multiselect("Select all the colon of target", main_data.columns)
                quant_val_list = []
                observations = st.number_input("Indicate how many quantile/percentile you want to check", min_value=0,
                                               key="num_of_observations")

                for num in range(observations):
                    quantile_val = st.number_input(f"Enter the quantile/percentile No: {num + 1} that you want to check",
                                                   min_value=0.00, max_value=100.00, key=f"quantile_{num}")
                    quant_val_list.append(quantile_val/100)

                if len(observatory_col) > 0 and len(quant_val_list) > 0:
                    my_quantiles = main_data[observatory_col].quantile(quant_val_list)
                    st.write(my_quantiles)
            else:
                st.write("Either your dataset is None or Empty")
        except Exception as e:
            st.write(f"{e}")

    # __________________________________________________________________________________________________________________

    def data_obs(self):

        df = FileManagement.data_clearing(self, self.data)

        if isinstance(df, pd.DataFrame) and df is not None and not df.empty:
            def_fin = df

            if st.checkbox("Check quantile/percentile", key="def_fin_not_sorted"):
                self.quartile_percentile(def_fin)

            action = st.selectbox("Select the action to apply on your delimited data",
                                  ['None', 'Data description', 'Data sorting', 'Data plot', 'Deep lensing'])

            if action == "Data description":
                st.subheader(f"Prime description of {self.subtitle}")
                st.write(def_fin.describe())

            elif action == "Data sorting":
                directions = []
                to_sort_by = st.multiselect("Select the column to sort_by", [col for col in def_fin.columns])
                for elem in to_sort_by:
                    direction = st.radio(f"Determine the direction for {elem}:", [None, "Ascending", "Descending"],
                                         key=f"{elem}_{to_sort_by.index(f"{elem}")}")
                    if direction == "Ascending":
                        directions.append(True)
                    elif direction == "Descending":
                        directions.append(False)
                if st.checkbox("process the sorting", key="data_sorting"):
                    try:
                        def_fin = def_fin.sort_values(by=to_sort_by, ascending=directions)
                        st.write(def_fin)
                        st.success("Sorted successfully !")

                        if st.checkbox("Check quantile/percentile", key="sorted_def_fin"):
                            self.quartile_percentile(def_fin)

                    except Exception as e:
                        st.write(f"{e}")

            elif action == "Data plot":
                column = st.selectbox("Select the column to plot", [None] + [col for col in def_fin.columns])
                chart_type = st.selectbox("Select the chart type", [None, "Bar chart", "Pie chart"])

                if chart_type == "Bar chart":
                    if column is not None:
                        st.subheader(f"Bar chart visualization of the *{column}* column in {self.subtitle}")
                        fig = plx.histogram(def_fin, x=def_fin[column], color=column)
                        st.plotly_chart(fig)

                elif chart_type == "Pie chart":
                    st.subheader(f"Pie chart visualization of the *{column}* column in {self.subtitle}")
                    display_mode = st.radio("Select a display mode:", [None, "Normal", "Pull maximum", "Pull minimum",
                                                                       "Specify x", "Gather some x"])

                    if display_mode == "Normal":
                        df_counts = def_fin[column].value_counts().reset_index()
                        df_counts.columns = [column, 'Number']
                        fig = plx.pie(df_counts, names=column, values='Number', title=f"{column} Pie chart")
                        st.plotly_chart(fig)

                    elif display_mode == "Pull maximum":
                        df_counts = def_fin[column].value_counts().reset_index()
                        df_counts.columns = [column, 'Number']
                        max_cat = df_counts.iloc[df_counts['Number'].idxmax()][column]
                        pull_values = [0.2 if cat == max_cat else 0 for cat in df_counts[column]]
                        couleur = sns.color_palette("pastel", len(df_counts))

                        fig, ax = plt.subplots(figsize=(9, 9))
                        ax.pie(df_counts['Number'], labels=df_counts[column], explode=pull_values, colors=couleur,
                               autopct='%1.1f%%')
                        ax.legend(df_counts[column], title=column, loc="lower left", fontsize=9)
                        ax.set_title(f"{column} Pie Chart")
                        st.pyplot(fig)

                    elif display_mode == "Pull minimum":
                        df_counts = def_fin[column].value_counts().reset_index()
                        df_counts.columns = [column, 'Number']
                        min_cat = df_counts.iloc[df_counts['Number'].idxmin()][column]
                        pull_values = [0.2 if cat == min_cat else 0 for cat in df_counts[column]]
                        couleur = sns.color_palette("pastel", len(df_counts))

                        fig, ax = plt.subplots(figsize=(9, 9))
                        ax.pie(df_counts['Number'], labels=df_counts[column], explode=pull_values, colors=couleur,
                               autopct='%1.1f%%')
                        ax.legend(df_counts[column], title=column, loc="lower left", fontsize=9)
                        ax.set_title(f"{column} Pie Chart")
                        st.pyplot(fig)

                    elif display_mode == "Specify x":
                        value = st.selectbox("Select the value to pull", [None]+[x for x in def_fin[column].unique()])
                        df_counts = def_fin[column].value_counts().reset_index()
                        df_counts.columns = [column, 'Number']
                        if value is not None:
                            max_index = df_counts[df_counts[column] == value].index[0]
                            explode_values = [0.2 if i == max_index else 0 for i in range(len(df_counts))]
                            palette = sns.color_palette("pastel")
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.pie(df_counts['Number'], labels=df_counts[column], explode=explode_values, colors=palette,
                                   autopct='%1.1f%%')
                            ax.set_title(f"{value}")
                            st.pyplot(fig)

                    elif display_mode == "Gather some x":
                        value = st.multiselect("Select the value to gather", [x for x in def_fin[column].unique()])
                        df_counts = def_fin[column].value_counts()
                        tot_sum = len(def_fin[column])
                        value_sum = 0
                        mem = {}
                        divs = st.columns([5, 3])

                        for col in value:
                            mem[col] = f"{((df_counts[col] / tot_sum) * 100).round(1)} %"
                            value_sum += df_counts[col]
                            df_counts = df_counts.drop(col)

                        with divs[0]:
                            if len(value) >= 2:
                                df_counts["Other"] = value_sum
                            df_counts = df_counts.reset_index()
                            df_counts.columns = [column, 'Number']
                            fig = plx.pie(df_counts, names=column, values='Number', title=f"{column} Pie chart")
                            st.plotly_chart(fig)

                        with divs[1]:
                            if len(mem) > 0:
                                st.subheader("-----------")
                                ref = pd.DataFrame(list(mem.items()), columns=["Category", "Rate"])
                                ref.title = f"Other---{((value_sum/tot_sum) * 100).round(1)} %"
                                st.write(ref.title, ref)

                    else:
                        st.write("Pick an element different from 'None'")

            elif action == "Deep lensing":

                strt_col = st.multiselect("Select all the column of target", [col for col in def_fin.columns])
                options = st.selectbox("Select an option", [None, "General look", "Particular look",
                                                            "Special look(object_non-string)"])

                if options is not None:
                    num_df = def_fin[[c for c in def_fin.columns if ptypes.is_numeric_dtype(def_fin[c]) is True]]
                    cat_df = def_fin[[c for c in def_fin.columns if ptypes.is_numeric_dtype(def_fin[c]) is False]]

                    if options == "General look":

                        for x in strt_col:
                            if x not in num_df:
                                num_df[x] = def_fin[x]
                            if x not in cat_df:
                                cat_df[x] = def_fin[x]

                        st.subheader("Numeric operations")

                        scr_col1, scr_col2 = st.columns([1, 5])

                        with scr_col1:
                            numeric_operations = [None, "count", "sum", "unique", "min", "max", "mean", "std", "mode",
                                                  "first", "last"]
                            operation = st.radio("Pick_up an operation for lensing", numeric_operations)

                        with scr_col2:

                            if operation:
                                try:
                                    if operation == "unique":
                                        result = num_df.groupby([x for x in strt_col]).apply(lambda y: y.nunique())
                                        with st.expander("See the 'Uniques':"):
                                            for x in def_fin:
                                                st.write(f"{x}: {def_fin[x].unique()}")

                                    elif operation == "mode":
                                        result = num_df.groupby([x for x in strt_col]).apply(lambda y: y.mode().iloc[0])
                                    else:
                                        result = num_df.groupby([x for x in strt_col]).agg(operation)
                                    st.write(result)

                                    if st.checkbox("Check quantile/percentile", key="result_numerical"):
                                        self.quartile_percentile(result)

                                    with st.expander("See the graph"):
                                        if operation != "mode":
                                            if result.empty:
                                                st.write("The provided data frame is empty.")
                                            else:
                                                my_fig = (plx.bar(result, y=result.columns, color="variable",
                                                          title=f"{operation}_special categorical plot",
                                                                  barmode="group"))
                                                st.plotly_chart(my_fig)
                                        else:
                                            st.write("No graph set for the 'mode'!")

                                    with st.expander("See description"):
                                        st.write(result.describe())

                                except Exception as e:
                                    st.write(f"Humm!...{e}")

                        st.subheader("Categorical/Object operations")

                        scr_col3, scr_col4 = st.columns([1, 5])

                        with scr_col3:
                            categorical_operations = [None, "count", "unique", "mode"]
                            operation = st.radio("Pick_up an operation for lensing", categorical_operations,
                                                 key='categorical_radio')

                        with scr_col4:
                            if operation:
                                try:
                                    if operation == "unique":
                                        result = cat_df.groupby(strt_col).apply(lambda y: y.nunique())
                                        with st.expander("See the 'Uniques':"):
                                            for x in cat_df:
                                                st.write(f"{x}: {def_fin[x].unique()}")
                                    elif operation == "mode":
                                        result = cat_df.groupby(strt_col).apply(lambda y: y.mode().iloc[0])
                                    else:
                                        result = cat_df.groupby(strt_col).agg(operation)
                                    st.write(result)

                                    if st.checkbox("Check quantile/percentile", key="result_category"):
                                        self.quartile_percentile(result)

                                    with st.expander("See the graph"):
                                        if operation != "mode":
                                            if result.empty:
                                                st.write("The provided data frame is empty.")
                                            else:
                                                my_fig = (plx.bar(result, y=result.columns, color="variable",
                                                          title=f"{operation}_special categorical plot",
                                                                  barmode="group"))
                                                st.plotly_chart(my_fig)
                                        else:
                                            st.write("No graph set for the 'mode'!")

                                    with st.expander("See description"):
                                        st.write(result.describe())

                                except Exception as e:
                                    st.write(f"Humm!...{e}")

                    elif options == "Particular look":

                        if len(def_fin.columns) > len(strt_col):

                            try:
                                columns_obj = st.multiselect("Select all the object_columns",
                                                             [col for col in def_fin.columns if col not in
                                                              strt_col])
                                num_operations = ["count", "sum",  "min", "max", "mean", "std"]
                                cat_operations = ["count"]

                                num_dic = {obj_col: num_operations for obj_col in columns_obj
                                           if obj_col in num_df.columns}
                                cat_dic = {obj_col: cat_operations for obj_col in columns_obj
                                           if obj_col in cat_df.columns}
                                num_dic.update(cat_dic)

                                prime_result = def_fin.groupby(strt_col).agg(num_dic)
                                tup = [(key, val) for key, lista in num_dic.items() for val in lista]

                                prime_result.columns = pd.MultiIndex.from_tuples(tup)

                                unique_result = (def_fin.groupby(strt_col).apply(lambda y: y[columns_obj].nunique()).
                                                 add_suffix('_unique'))
                                mode_result = (def_fin.groupby(strt_col).apply(lambda y: y[columns_obj].mode().iloc[0]).
                                               add_suffix('_mode'))

                                for col in columns_obj:
                                    if col in num_df:
                                        rank = prime_result.columns.get_loc((col, "std"))
                                        prime_result.insert(loc=rank+1, column=(col, "unique"),
                                                            value=unique_result[f"{col}_unique"])
                                        prime_result.insert(loc=rank+2, column=(col, "mode"),
                                                            value=mode_result[f"{col}_mode"])

                                    elif col in cat_df:
                                        rank = prime_result.columns.get_loc((col, "count"))
                                        prime_result.insert(loc=rank+1, column=(col, "unique"),
                                                            value=unique_result[f"{col}_unique"])
                                        prime_result.insert(loc=rank+2, column=(col, "mode"),
                                                            value=mode_result[f"{col}_mode"])

                                st.write(prime_result)
                                u_col, m_col = st.columns(2)
                                with u_col:
                                    st.subheader('N_unique_values')
                                    st.text("Number of individual values occurred")
                                    st.write(unique_result)

                                with m_col:
                                    st.subheader('Mode_values')
                                    st.text("Values with the most occurrence")
                                    st.write(mode_result)

                            except Exception as e:
                                st.write(f"{e}")

                        else:
                            st.write("There's no column for a particular look")

                    elif options == "Special look(object_non-string)":

                        st.write("This section is specially made for date/time data_type sets."
                                 "So, the required key should be a column which has date/time or object as data_type")

                        key_elem = st.multiselect("Mention the key_value", strt_col, key="key_elem")
                        index_elem = st.multiselect("Mention the grouping_by value", [x for x in strt_col if x not
                                                    in key_elem], key="index_elem")

                        if st.checkbox("Proceed"):
                            try:
                                op_col, val_col = st.columns([2, 5])
                                with op_col:
                                    operatora = ["count", "sum", "min", "max", "mean", "std",
                                                 "mode", "unique", "nunique", "first", "last"]
                                    operatora_ok = []

                                    for x in operatora:
                                        if st.checkbox(x, key=f"operatora_{operatora.index(x)}"):
                                            operatora_ok.append(x)

                                    resetting_index = st.radio("Reset the dataset:", ["no", "yes"], key="resetting")

                                with val_col:
                                    if operatora_ok is not None and key_elem is not None and index_elem is not None:
                                        returning = def_fin.groupby(index_elem)[key_elem].agg(operatora_ok)

                                        if resetting_index == "yes":
                                            returning = returning.reset_index()

                                        st.success("Successful grouping !")
                                        st.write(returning)

                                        if st.checkbox("Check quantile/percentile", key="returning"):
                                            self.quartile_percentile(returning)
                                    else:
                                        st.write("Pick an operation")

                            except Exception as e:
                                st.error(f"Something went bad: {e}")

        else:
            st.write("There's no DataFrame found yet")

    # __________________________________________________________________________________________________________________
    # Machine Learning Models

    def encoder(self, data, colon_list, keypass, manually=False):
        my_data = None
        if manually is True:
            n = 0
            mapping_in = {}
            mapping_glob = {}

            for col in colon_list:
                for category in data[col].unique():
                    code = st.number_input(f"Set a number_code for {category}", min_value=0, key=n)
                    n += 1
                    mapping_in[category] = code
                mapping_glob[col] = mapping_in
                if st.checkbox("Validate the manual process", key=f"manual_{keypass}"):
                    data[col] = data[col].replace(mapping_in)
                    my_data = data
            self.memory_diction = mapping_glob

        else:
            if st.checkbox("Validate the automatic process", key=f'auto_{keypass}'):
                data_encoded = pd.get_dummies(data[colon_list], columns=colon_list)
                my_data = data_encoded
        return my_data

    def check_object(self, data, my_list, ref_list):
        result_list = []
        for c_ in my_list:
            if (data[c_].dtype == object or pd.api.types.is_datetime64_any_dtype(data[c_]) is True) and c_ in ref_list:
                result_list.append(c_)
        return result_list

    def concat(self, data1=None, data2=None):

        if data1 is not None and data2 is not None:
            glob_dat = pd.concat([data1, data2], axis=1)
            return glob_dat
        else:
            if data1 is not None and data2 is None:
                return data1
            elif data1 is None and data2 is not None:
                return data2

    def correlation(self, data, index_ref, column_ref):
        my_sum = 0
        for i_value in index_ref:
            for x_value in column_ref:
                my_sum += data.loc[i_value, x_value]

        correlation = my_sum/(len(index_ref) * len(column_ref))
        styled_data = data.style.applymap(lambda val: "background-color: orange"
                                          if val in data.loc[index_ref, column_ref].values else "")

        return correlation, styled_data

    def explanation(self, concern):
        text = None
        if concern == "correlation":
            text = st.write("""
                    This correlation should not hasten our decision... 
                    Despite its value and the conclusion it might suggest, it is recommended to proceed 
                    with the in target model to further verify the dependencies of the variables.

                    Interpretations:
                    ________________
                                >If the correlation is close to 0 → This means there is no or very little 
                    linear relationship between the variables. They evolve independently of each other.

                                >If the correlation is close to +1 → Indicates a strong positive relationship:
                    when one increases, the other tends to increase as well.

                                >If the correlation is close to -1 → Indicates a strong negative relationship: 
                    when one increases, the other decreases.

                    A correlation close to 0 does not always mean that there is no relationship at all, 
                    but rather that there is no clear linear relationship. Other types of dependencies 
                    may exist, such as non-linear relationships that can be detected using advanced models.""")

        elif concern == "test_size":
            text = st.write("""The test size defines the distribution between the training data, and the test data.
                     For example, a test_size of 0.3, will split the data into 70% of training data and 30% of test_data.

                     > Higher test_size : Higher precision in evaluation, low robustness of the model.
                     > Lower test_size : Lower precision in evaluation, higher robustness of the model.""")

        elif concern == "Model_interpret":
            text = st.write(""" 
                                A) LINEAR REGRESSION
                                  __________________
            
                                1. The intercept(constant):
                                            
                                  The intercept represents the value of the dependent variable (Y) when all 
                                  predictor variables (X) are equal to zero.
                                  
                                  > If the intercept is positive, it means that Y starts at a certain value even when 
                                  there are no predictors.
                                  > If the intercept is negative, it suggests that Y starts below zero in the absence of
                                   predictors.
                                _______________________
                                2. The MSE(error):
                                  
                                  MSE is a key metric that measures how well the model’s predictions 
                                  match the actual values.
                                  It is calculated as the average squared difference between the actual values (y_test) 
                                  and the predicted values (y_predict).
                                  
                                 🔹 Low MSE → The model makes accurate predictions.
                                 🔹 High MSE → The model has large errors and may need improvement.
                                _______________________
                                3. The R-Score( R²):
                                  
                                   R² ranges from 0 to 1 (or 0% to 100% when expressed as a percentage).
                                   
                                   A high R² (close to 1) means the model explains most of the variance in the data.
                                   A low R² (close to 0) means the model explains little of the variance—meaning 
                                   it's not capturing important patterns.
                                _______________________ 
                                4. The P-Value:
                                  
                                   In a linear regression model, the p-value is a crucial statistical metric used to 
                                   determine whether a predictor variable (X) significantly influences the dependent 
                                   variable (Y).
                                   
                                   > A small p-value (typically < 0.05) suggests strong evidence that the predictor does
                                    have an impact on Y, so we reject the null hypothesis.
                                   > A large p-value (> 0.05) suggests weak evidence that the predictor influences Y, 
                                   meaning it might not be a significant variable in the model.
                                   
                                   Thus, p-values help decide which predictors are truly impactful in the model.
                                   
                                   How P-Values Relate to Predictor Coefficients:
                                   
                                   Each predictor in the linear model (X1, X2, X3, ...) has: 
                                   ✔ A coefficient (β) → Indicates how much Y changes per unit increase in X.
                                   ✔ A p-value → Indicates whether this coefficient is statistically significant.
                                   💡 Key interpretation:
                                   - If a predictor has a large coefficient with a high p-value, it means the effect 
                                    exists but is not statistically significant (it could be due to randomness).
                                    
                                   - If a predictor has a large coefficient with a low p-value, it means the variable 
                                    significantly impacts Y.
                                    
                                   - If both the coefficient and p-value are high, the predictor has little effect and 
                                    is insignificant even if the change is high.
                                -------------------------------------------------------------------
                                B) LOGISTIC REGRESSION
                                  ____________________
                                    
                                1. The Accuracy:
                                 
                                   Accuracy is for classification models
                                   Accuracy measures the percentage of correct predictions for categorical labels 
                                   (e.g., "cat" vs. "dog").
                                   Linear regression predicts numerical values, so accuracy doesn’t make sense 
                                   for evaluating its performance.
                                   
                                   > Example of Incorrect Accuracy Usage
                                   If a model predicts house prices and the actual prices are [100k, 200k, 300k], 
                                   accuracy would attempt to check how many predictions are exactly correct—which 
                                   is unrealistic since regression outputs continuous values.
                                _______________________
                                2. The Confusion Matrix:
                                   
                                   The confusion matrix is a table that summarizes the predictions of the model versus 
                                   the actual classifications. It includes:
                                    - True Positives (TP): Correctly predicted positive cases.
                                    - True Negatives (TN): Correctly predicted negative cases.
                                    - False Positives (FP): Incorrectly predicted positive cases (Type I Error).
                                    - False Negatives (FN): Incorrectly predicted negative cases (Type II Error).
                                    
                                    🔹 Interpretation:
                                    A high number of TP and TN means your model is performing well, 
                                    whereas high FP and FN indicate errors that need to be reduced.
                                _______________________
                                3. Precision, Recall, and F1-score:
                                   
                                    - Precision: Measures how many of the predicted positive cases are actually positive
                                    - Recall (Sensitivity): Measures how well the model identifies actual positive cases
                                    - F1-score: A balanced measure of Precision and Recall. 
                                    
                                    🔹 Interpretation:
                                    - High Precision means fewer false positives (important for fraud detection).
                                    - High Recall means fewer false negatives (important in medical diagnoses).
                                    - F1-score helps in finding the right balance when both precision and recall 
                                    are important.
                                _______________________
                                4. ROC Curve and AUC:
                                   
                                    - The Receiver Operating Characteristic (ROC) curve shows the trade-off between 
                                    sensitivity (TP rate) and false positive rate (FP rate) across different threshold
                                     values.
                                    - Area Under the Curve (AUC): Measures the overall ability of the model to 
                                    distinguish between classes.
                                
                                    🔹 Interpretation:
                                    - AUC close to 1 → Excellent model.
                                    - AUC around 0.5 → No better than random guessing.
                                    - AUC < 0.5 → Model is performing worse than random.
                                _______________________
                                5. Log-loss (Logarithmic Loss)
                                  
                                   This metric evaluates the accuracy of the probability predictions of the model.
                                
                                   🔹 Interpretation:
                                   - A lower log-loss indicates better-calibrated probability predictions.
                                   - A higher log-loss means the model is making incorrect or uncertain predictions.
                                    """)

        return text

    def variable_prep(self, sub_data, ref_list, ref_name, key_time):
        cate_list = FileManagement.check_object(self, sub_data, sub_data.columns, ref_list)
        num_list = [xcol for xcol in ref_list if xcol not in cate_list]
        data_num = sub_data[num_list]
        whole_data = None
        try:
            st.write("-----------------------------")
            st.subheader(f"{ref_name}_variable(s)")
            memo_dict, dat1, dat2 = None, None, None

            if len(cate_list) > 0:
                st.write(f"""The following variable(s): {cate_list} is/are object values and need an encoding !
                             This can be done manually or automatically.
                             1. Manually: you provide a relative code(ie. there will be graduation among values)
                              to a categorical value of the variable
                             2. Automatically, the program assign to each categorical value a non relative 
                              binary code(made of  0(s) and(1s)""")

                no_touch = []
                diviser = st.columns([5, 5, 3])

                with diviser[0]:
                    take = st.radio("", ["Pass", "Encode some variable manually"], key=f"radio{key_time}")
                    if take == "Encode some variable manually":
                        manualist = st.multiselect("Select all the data you want to encode manually", cate_list,
                                                   key=f"multiselect{key_time}")
                        no_touch = manualist

                        if manualist is not None:
                            dat1 = FileManagement.encoder(self, sub_data[manualist], manualist, key_time, manually=True)
                        else:
                            st.write("Waiting for section completion...")

                with diviser[1]:
                    if st.checkbox("Encode all rest automatically", key=f"automatic_{key_time}"):
                        autolist = [x for x in cate_list if x not in no_touch]
                        if autolist is not None:
                            dat2 = FileManagement.encoder(self, sub_data[autolist], autolist, key_time, manually=False)

                if len(data_num.columns) >= 2:
                    with diviser[2]:
                        if st.checkbox("Normalize numerical values", key=f"norm{key_time}"):
                            norm_typ = st.radio("", [None, "Standardization(Z-score)", "MinMax_Scaler"],
                                                key=f"norm_typ{key_time}")
                            if norm_typ == "Standardization(Z-score)":
                                scaler = StandardScaler()
                                data_num = scaler.fit_transform(data_num)
                            elif norm_typ == "MinMax_Scaler":
                                scaler = MinMaxScaler()
                                data_num = scaler.fit_transform(data_num)
            else:
                st.write("All variable(s) are numeric in this section.")
                if len(data_num.columns) >= 2:
                    diviser = st.columns([5, 5, 3])
                    with diviser[1]:
                        if st.checkbox("Normalize numerical values", key=f"norm{key_time}"):
                            norm_typ = st.radio("", [None, "Standardization(Z-score)", "MinMax_Scaler"],
                                                key=f"norm_typ{key_time}")
                            if norm_typ == "Standardization(Z-score)":
                                scaler = StandardScaler()
                                data_num = scaler.fit_transform(data_num)
                            elif norm_typ == "MinMax_Scaler":
                                scaler = MinMaxScaler()
                                data_num = scaler.fit_transform(data_num)

            if st.checkbox("Go on", key=f"checkbox{key_time}"):
                data_cat = FileManagement.concat(self, dat1, dat2)
                whole_data = FileManagement.concat(self, data_num, data_cat)
                memo_dict = self.memory_diction

                st.success(f"The {ref_name} is/are encoded with success !")
                with st.expander("Check the encodings"):
                    st.write(data_cat, memo_dict)
                with st.expander(f"See the whole data_{ref_name}"):
                    st.write(whole_data)

            return whole_data
        except Exception as e:
            st.write(f"{e}")

    def data_test(self, inner_dt, data_train, ref_list, ref_dic):
        auto_dt = None
        manu_dt = None
        num_dt = None

        for col_name in inner_dt.columns:
            st.write(f"{col_name}:")
            if col_name in FileManagement.check_object(self, inner_dt, inner_dt.columns, ref_list):
                targ_val = st.selectbox("Pick the target value", [x for x in inner_dt[col_name].unique()],
                                        key=f"{col_name}_check")
                if col_name in ref_dic.keys():
                    dat1_manu = {col_name: [ref_dic[col_name][targ_val]]}
                    df = pd.DataFrame(dat1_manu)
                    manu_dt = FileManagement.concat(self, df, manu_dt)

                else:
                    dat1_auto = {col_name: inner_dt[col_name].unique()}
                    df = pd.DataFrame(dat1_auto)
                    df = pd.get_dummies(df, columns=[col_name])
                    df_filtered = df[df[f"{col_name}_{targ_val}"] == 1]
                    df_filtered = df_filtered.reset_index(drop=True)
                    auto_dt = FileManagement.concat(self, df_filtered, auto_dt)

            else:
                targ_val = st.number_input("Enter the value:", key=f"{col_name}_check")
                dat2 = {col_name: [targ_val]}
                df = pd.DataFrame(dat2)
                num_dt = FileManagement.concat(self, df, num_dt)

        if st.checkbox("Submit", key="submit"):
            data_test = FileManagement.concat(self, manu_dt, auto_dt)
            data_test = FileManagement.concat(self, data_test, num_dt)
            data_test = data_test[data_train.columns]
            st.write(data_test)

            return data_test

    def build_function(self, variable, coefficient, **kwargs):
        n = 1
        func = ""
        constant = kwargs.get("constant")
        error = kwargs.get("error")
        coefficient_list = coefficient[0].tolist()
        variable_in_x = [f"X{n+1}" for n in range(len(coefficient_list))]

        for i in range(len(variable_in_x)):
            func = func + f"+ ({coefficient_list[i]}){variable_in_x[i]} "

        script = dict(zip(variable, coefficient_list))
        script_total = {x: {k: script[k]} for x, k in zip(variable_in_x, script)}

        if constant is not None:
            func = f"({constant}) " + func
            script_total["constant(intercept)"] = constant
        if error is not None:
            func = func + f"+ {error}"
            script_total["error"] = error

        return func, script_total

    def linear_model_apply(self, data, x_dt, y_dt, target_model, **entry):
        x_var = data[x_dt]
        y_var = data[y_dt]

        x_check_num = x_var.select_dtypes(exclude=['number']).empty
        y_check_num = y_var.select_dtypes(exclude=['number']).empty
        dt_split = st.number_input("Provide the test_size value", min_value=0.000, max_value=1.000, key="model")
        with st.expander("Test_size explanation"):
            FileManagement.explanation(self, "test_size")

        if st.checkbox("proceed", key=" model_proceed"):

            status = st.empty()
            progress_bar = st.progress(0)
            for i in range(101):
                time.sleep(0.08)
                progress_bar.progress(i)
                status.text(f"Processing : {i}%")

            x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size=dt_split, random_state=42)

            summary_table = "Not available for Logistic Regression"
            r_score = "Not available for Logistic Regression"

            accuracy = "Not available for Linear Regression"
            conf_m = "Not available for Linear Regression"
            precision = "Not available for Linear Regression"
            recall = "Not available for Linear Regression"
            f1_sc = "Not available for Linear Regression"

            fig = None
            err = fig

            if len(y_var.columns) > 1:
                # categorical(logistical model)

                y_train = y_train.idxmax(axis=1)
                y_train = y_train.fillna(0)
                y_test = y_test.idxmax(axis=1)
                y_test = y_test.fillna(0)

                model = target_model
                model.fit(x_train, y_train)
                y_predict = model.predict(x_test)
                y_prob = model.predict_proba(x_test)

                try:
                    accuracy = accuracy_score(y_test, y_predict)
                except Exception as e:
                    accuracy = e
                try:
                    conf_m = confusion_matrix(y_test, y_predict)
                except Exception as e:
                    conf_m = e
                try:
                    precision = precision_score(y_test, y_predict, average='weighted')
                except Exception as e:
                    precision = e
                try:
                    recall = recall_score(y_test, y_predict, average='weighted')
                except Exception as e:
                    recall = e
                try:
                    f1_sc = f1_score(y_test, y_predict, average='weighted')
                except Exception as e:
                    f1_sc = e

                try:
                    classes = np.unique(y_test)

                    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

                    # Création du graphique with Seaborn
                    fig, ax = plt.subplots()
                    sns.set_style("darkgrid")  # Style Seaborn

                    for i in range(len(classes)):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                        auc_score = auc(fpr, tpr)
                        if not np.isnan(auc_score):  # check that there's no AUC = nan
                            sns.lineplot(x=fpr, y=tpr, ax=ax, label=f"Class {classes[i]} (AUC={auc_score:.2f})")

                    ax.set_xlabel("False positive rate (FPR)")
                    ax.set_ylabel("True positive rate (TPR)")
                    ax.set_title("ROC_curve")
                    ax.legend()
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

                except Exception as e:
                    err = f"Kindly check the following error: {e}"

            else:
                # numerical(regression model)

                model = target_model
                model.fit(x_train, y_train)
                y_predict = model.predict(x_test)
                try:
                    r_score = r2_score(y_test, y_predict)
                except Exception as e:
                    r_score = e

            st.success("Model successfully built !")

            with st.expander("Model info"):
                if x_check_num is True and y_check_num is True:
                    coefficients = np.round(model.coef_, 2)
                    constant = np.round(model.intercept_, 2)
                    mse = round(mean_squared_error(y_test, y_predict), 2)
                    function_returns = FileManagement.build_function(self, x_dt, coefficients, constant=constant[0],
                                                                     error=mse)
                    divider = st.columns([5, 3])
                    with divider[0]:
                        st.write("The generative function of the model is :")
                        st.success(function_returns[0])
                    with divider[1]:
                        st.write("Where the parameter's reference is:")
                        st.write(function_returns[1])

                st.write("The R-score of the model is :")
                st.success(r_score)

                st.write("The statistical summary_table of the model is :")
                st.write(summary_table)

                st.write("The Accuracy of the model is :")
                st.success(accuracy)

                st.write("The Confusion-matrix of the model is :")
                st.write(conf_m)

                st.write("The Precision-score of the model is :")
                st.success(precision)

                st.write("The Recall-score(sensitivity) of the model is :")
                st.success(recall)

                st.write("The F1-score of the model is :")
                st.success(f1_sc)

                if fig is not None:
                    st.write("The following graphic is representative of:")
                    st.pyplot(fig)
                else:
                    st.success(err)

            with st.expander("Statistic interpretations"):
                FileManagement.explanation(self, "Model_interpret")

            st.write("----------------------------------------")

            if st.checkbox("Test your data"):

                x_entry = FileManagement.data_test(self, entry["inner_dt_predictor"], entry["the_train"],
                                                   entry["ref_list"], entry["ref_dico"])
                if x_entry is not None:

                    with st.spinner("treating..."):
                        y_output = model.predict(x_entry)
                        for i in range(51):
                            time.sleep(0.08)

                    st.write(f"From the provided data, the prediction of {entry["foresee"]} is :")
                    st.success(y_output[0])
                else:
                    st.write("Submit the predictor's value...")

    def linear_model(self, trgtdt):
        if trgtdt is not None:
            dt = trgtdt
            lines = len(dt)
            variables = (lines - (lines % 20)) / 20

            with st.expander("See data_info for modelisation"):
                st.write(data_info(dt))

                if lines >= 20:
                    st.success(f"The number of data is sufficient for a linear model"
                               f"with {variables} variable(s) or predictor(s)")

                else:
                    st.error("The numer of data is less than 20..., very low to perform any linear model training")

            predictor = st.multiselect("Select the predictor_variable(s)", [col for col in dt.columns])
            if len(predictor) > variables:
                with st.expander("ALERT‼️"):
                    st.error(f"The quantity of data can only handle a model of {variables} predictor(s) in maximum..."
                             f"Going above may affect the quality of the model")

            predicted = st.multiselect("Select the predicted_variable",
                                       [col for col in dt.columns if col not in predictor])

            if st.checkbox("Continue..."):
                if len(predictor) == 0 or len(predicted) == 0:
                    st.error("There should be missing data either in predictor or predicted section !!")
                else:
                    sub_trgtdt = dt[predictor + predicted]
                    whole_data_predictor = FileManagement.variable_prep(self, sub_data=sub_trgtdt, ref_list=predictor,
                                                                        ref_name="Predictor(s)", key_time=1)
                    whole_data_predicted = FileManagement.variable_prep(self, sub_data=sub_trgtdt, ref_list=predicted,
                                                                        ref_name="Predicted(s)", key_time=2)

                    if whole_data_predictor is not None and whole_data_predicted is not None:
                        sub_trgtdt = FileManagement.concat(self, whole_data_predictor, whole_data_predicted)
                        corr_matrix = sub_trgtdt.corr(method="pearson")

                        my_return = FileManagement.correlation(self, corr_matrix, whole_data_predictor.columns,
                                                               whole_data_predicted.columns)

                        la_correlation = my_return[0]
                        colored_dt = my_return[1]

                        st.write("--------------------------------")
                        with st.expander("Check the correlation info"):
                            st.write("The Correlation Matrix")
                            st.write(colored_dt)
                            st.write("The overall correlation between the predictor and the predicted values is:")
                            st.success(la_correlation)
                            FileManagement.explanation(self, "correlation")

                        st.write("____________________________________________________________________________________")

                        if st.checkbox("Kick-Start a linear model", key="Go_now"):

                            the_x = whole_data_predictor.columns
                            the_y = whole_data_predicted.columns

                            condition = whole_data_predicted.select_dtypes(exclude=['number']).empty

                            if condition is True:
                                st.subheader("This is a Linear Regression model.")
                                FileManagement.linear_model_apply(self, sub_trgtdt, the_x, the_y, LinearRegression(),
                                    inner_dt_predictor=dt[predictor], the_train=whole_data_predictor,
                                    ref_list=predictor, ref_dico=self.memory_diction, foresee=predicted[0])

                            else:
                                st.subheader("This is a Logistic Regression model.")
                                FileManagement.linear_model_apply(self, sub_trgtdt, the_x, the_y, LogisticRegression(),
                                    inner_dt_predictor=dt[predictor], the_train=whole_data_predictor,
                                    ref_list=predictor, ref_dico=self.memory_diction, foresee=predicted[0])



