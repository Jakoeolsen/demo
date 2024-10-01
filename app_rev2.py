import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import requests

st.set_page_config(layout="wide")

# Function to make the API call and return the DataFrame
def fetch_data(tso, start_date, end_date):
    url = "https://test-publicationtool.jao.eu/nordic/api/data/finalComputation"
    headers = {"Authorization": "Bearer ***token***"}  # Replace ***token*** with your actual token
    take = 4000000

    tso_mapping = {
        "ENERGINET": "10X1001A1001A248",
        "FINGRID": "10X1001A1001A264",
        "STATNETT": "10X1001A1001A38Y",
        "SVENSKE KRAFTNÄT": "10X1001A1001A45C"  # Assuming this is the correct code for SVENSKE KRAFTNÄT
    }

    # Convert the dates to UTC string format
    from_utc = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    to_utc = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    params = {
        "NonRedundant": True,
        "Filter": f'{{"Tso":["{tso_mapping[tso]}"]}}',
        "Skip": 0,
        "Take": take,
        "FromUtc": from_utc,
        "ToUtc": to_utc
    }

    # Make the GET request
    response = requests.get(url, headers=headers, params=params)

    # Check the response status and handle accordingly
    if response.status_code >= 200 and response.status_code < 300:
        data = response.json()['data']  # FROM JSON FORMAT TO DATA FRAME
        df = pd.DataFrame(data)
        df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'])

        return df
    else:
        error_message = f"HTML query returned an unexpected response. Status code: {response.status_code}. Response text: {response.text}"
        st.error(error_message)
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Function for the main page
def main_page():

    st.title("Temporary Dashboard for Go-Live")

    # Date range selection
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 4))
    end_date = st.date_input("End Date", value=datetime(2024, 1, 7))

    # TSO filter selection
    filter_option = st.selectbox("Choose a filter", ["ENERGINET", "STATNETT", "SVENSKE KRAFTNÄT", "FINGRID"])

    if st.button("Run API Call"):
        # Fetch data using the API
        df = fetch_data(filter_option, start_date, end_date)

        if not df.empty:
            # Store the DataFrame in session state
            st.session_state['dataframe'] = df
            st.success("DataFrame created and stored in session state! DATA Preview:")
            st.success(df.head())

        else:
            st.warning("No data returned from the API.")

# Function for the plot page
def plot_page():
    st.title("Plot Page")

    # Check if the DataFrame is available in session state
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']

        # Scroll bar to select a unique cnecName
        selected_cnecName = st.selectbox("Select a cnecName", df['cnecName'].unique())

        # Filter the DataFrame based on the selected cnecName
        filtered_df = df[df['cnecName'] == selected_cnecName]

        # Scroll bar to select columns for plotting
        columns_to_plot = st.multiselect(
            "Select columns to plot",
            options=['ram', 'minFlow', 'u', 'imax', 'fmax', 'frm', 'fref', 'fall', 'fnrao', 'amr', 'aac', 'iva'],
            default=['ram']  # Default selection can be modified
        )

        if not columns_to_plot:
            st.warning("Please select at least one column to plot.")
        else:
            # Plot the selected columns over time using Plotly
            fig = px.line(
                filtered_df,
                x='dateTimeUtc',
                y=columns_to_plot,
                title=f"Plot for {selected_cnecName}",
                labels={'dateTimeUtc': 'Time', 'value': 'Value'}
            )

            st.plotly_chart(fig)

        # Datetime selection
        selected_date = st.selectbox("Select a date", df['dateTimeUtc'].dt.date.unique())

        # Filter the DataFrame based on the selected date
        filtered_df = df[df['dateTimeUtc'].dt.date == selected_date]

        selected_time = st.selectbox("Select a time", filtered_df['dateTimeUtc'].dt.time.unique())

        # Further filter based on the selected time
        filtered_df_time = filtered_df[filtered_df['dateTimeUtc'].dt.time == selected_time]

        # Display a table with CNECs and RAM values for the selected timestamp, sorted by RAM value
        st.subheader(f"CNECs and RAM Values for {selected_date} {selected_time}")

        sorted_df = filtered_df_time[['cnecName','ptdf_DK1','ptdf_DK1_CO','ptdf_DK1_DE','ptdf_DK1_KS','ptdf_DK1_SK','ptdf_DK1_SB','ptdf_DK2','ptdf_DK2_KO','ptdf_DK2_SB','ptdf_FI','ptdf_FI_EL','ptdf_FI_FS','ptdf_NO1','ptdf_NO2','ptdf_NO2_ND','ptdf_NO2_SK','ptdf_NO2_NK','ptdf_NO3','ptdf_NO4','ptdf_NO5','ptdf_SE1','ptdf_SE2','ptdf_SE3','ptdf_SE3_FS','ptdf_SE3_KS','ptdf_SE3_SWL','ptdf_SE4','ptdf_SE4_BC','ptdf_SE4_NB','ptdf_SE4_SP','ptdf_SE4_SWL','ram']].drop_duplicates().sort_values(by='ram', ascending=False)
        st.dataframe(sorted_df, use_container_width=True)

    else:
        st.warning("No DataFrame found in session state. Please go to the Main Page and initiate the API call.")

# Function for the flow visualization map page
def map_page():
    st.title("Flow Visualization Map")

    # Date selection
    selected_date = st.date_input("Select a date", value=datetime.now().date())

    # Hour picker for selecting a specific hour (with one-hour resolution)
    hours = [f"{h:02d}:00" for h in range(24)]  # List of hours from 00:00 to 23:00
    selected_hour = st.selectbox("Select an hour", options=hours)

    # Combine the selected date and hour
    selected_datetime = datetime.combine(selected_date, datetime.strptime(selected_hour, "%H:%M").time())

    st.write(f"Visualizing data for datetime: {selected_datetime}")

    # Coordinates and flow data (You can modify these values based on your data)
    COORDINATES = {
        'DK1': [56.419776, 8.289597],
        'DK2': [55.326372, 11.930202],
        'SE4': [55.823288, 14.089607],
        'SE3': [60.043766, 15.180170],
        'SE2': [64.988753, 17.480684],
        'SE1': [67.375872, 21.217141],
        'NO4': [69.806297, 20.479831],
        'NO3': [63.455267, 10.515242],
        'NO1': [60.422174, 11.083172],
        'NO2': [58.209955, 7.210923],
        'NO5': [61.127959, 6.126693],
        'FI':  [62.469646, 26.675430],  # New Finland location
    }

    net_positions = {
        'DK1': 100,   # Example net position
        'DK2': -50,
        'SE4': 30,
        'SE3': -80,
        'SE2': 20,
        'SE1': 10,
        'NO4': -40,
        'NO3': 50,
        'NO1': -30,
        'NO2': 25,
        'NO5': -10,
        'FI':  60,    # Example net position for Finland
    }

    flows = {
        ('DK1', 'DK2'): 50,
        ('DK1', 'SE3'): 70,
        ('DK1', 'NO2'): -30,
        ('DK2', 'SE4'): 45,
        ('SE4', 'SE3'): 55,
        ('SE3', 'NO1'): -25,
        ('SE3', 'SE2'): 65,
        ('NO1', 'NO2'): 35,
        ('NO1', 'NO3'): -45,
        ('NO1', 'NO5'): 40,
        ('NO2', 'NO5'): -20,
        ('NO5', 'NO3'): 50,
        ('NO3', 'SE2'): -60,
        ('NO3', 'NO4'): 75,
        ('NO4', 'SE2'): 80,  # Corrected flow
        ('SE1', 'NO4'): 55,  # New flow
        ('SE1', 'SE2'): 65,  # New flow
        ('SE1', 'FI'):  40,  # New flow
        ('FI', 'SE3'):  50,  # New flow
        ('FI', 'NO4'):  45,  # New flow
    }

    arrow_scale = 5.0
    fig = go.Figure()

    def add_arrow(fig, source, target, flow_value, arrow_scale):
        source_coord = np.array(COORDINATES[source])
        target_coord = np.array(COORDINATES[target])
        
        if flow_value < 0:
            source_coord, target_coord = target_coord, source_coord
            flow_value = abs(flow_value)
        
        line_width = flow_value * arrow_scale / max(flows.values())
        
        fig.add_trace(go.Scattermapbox(
            lon=[source_coord[1], target_coord[1]],
            lat=[source_coord[0], target_coord[0]],
            mode='lines',
            line=dict(width=line_width, color='black'),
            opacity=0.7,
            hoverinfo='none'
        ))

        v = target_coord - source_coord
        v /= np.linalg.norm(v)

        l = 0.15 * arrow_scale
        w = 0.1 * arrow_scale
        
        u = np.array([-v[1], v[0]])

        P = target_coord - l * v
        S = P - w * u
        T = P + w * u

        fig.add_trace(go.Scattermapbox(
            lon=[S[1], target_coord[1], T[1], S[1]],
            lat=[S[0], target_coord[0], T[0], S[0]],
            mode='lines',
            fill='toself',
            fillcolor='green',
            line_color='green',
            hoverinfo='none'
        ))

        mid_point_lat = (source_coord[0] + target_coord[0]) / 2
        mid_point_lon = (source_coord[1] + target_coord[1]) / 2

        fig.add_trace(go.Scattermapbox(
            lon=[mid_point_lon],
            lat=[mid_point_lat],
            mode='text',
            text=[f"{flow_value:.1f} MW"],
            textfont=dict(size=14, color='blue'),
            textposition="middle center",
            hoverinfo='text',
            hovertext=f"{source} -> {target}: {flow_value:.1f} MW"
        ))

    for (source, target), flow_value in flows.items():
        add_arrow(fig, source, target, flow_value, arrow_scale)

    for region, coord in COORDINATES.items():
        fig.add_trace(go.Scattermapbox(
            lon=[coord[1]],
            lat=[coord[0]],
            mode='markers+text',
            marker=dict(size=20, color='red'),
            text=[f"{region}: {net_positions[region]}"],
            textposition="bottom center",
            textfont=dict(size=18, color='black', family='Arial Bold'),
            hoverinfo='text'
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=4,
            center=dict(lat=61, lon=15),
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        width=800,
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["API DATE SELECTOR", "CNEC PLOT PAGE", "Flow Visualization Map"])

# Render the selected page
if page == "API DATE SELECTOR":
    main_page()
elif page == "CNEC PLOT PAGE":
    plot_page()
elif page == "Flow Visualization Map":
    map_page()

