import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import requests
from lxml import etree

st.set_page_config(layout="wide")

# Example API endpoint and key


# Function to make the API call and return the DataFrame


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)




def fetch_data(start_date, end_date):
    headers = {"Authorization": "Bearer ***f{token}***"}  # Replace ***token*** with your actual token
    take = 4000000

    from_utc = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    to_utc = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    params = {
        "NonRedundant": True,
        "Filter": '{}',  # ENERGINET 10X1001A1001A248 / FILTER FOR TSO
        "Skip": 0,
        "Take": take,
        "FromUtc": from_utc,
        "ToUtc": to_utc
    }

    # Make the GET request
    response = requests.get(URL1, headers=headers, params=params)
    response_golive = requests.get(URL2, headers=headers, params=params)

    if response.status_code >= 200 and response.status_code < 300:
        data = response.json()['data'] # FROM JSON FORMAT TO DATA FRAME
        df = pd.DataFrame(data)            
        df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'], utc=True)
        df['dateTimeCET'] = df['dateTimeUtc'].dt.tz_convert('CET')
    
    else:
        error_message = f"HTML query returned an unexpected response. Status code: {response.status_code}. Response text: {response.text}"
        raise Exception(error_message)
    
    
    if response_golive.status_code >= 200 and response_golive.status_code < 300:
        data = response_golive.json()['data'] # FROM JSON FORMAT TO DATA FRAME
        df_go_live = pd.DataFrame(data)
    
    else:
        error_message = f"HTML query returned an unexpected response. Status code: {response.status_code}. Response text: {response.text}"
        raise Exception(error_message)

    #################################################################################################################################
    
    ## DELETE WHEN GO LIVE  ####
    
    #################################################################################################################################
    
    df_go_live = df.copy()
    # Generate shadow price column
    np.random.seed(42)  # For reproducibility
    shadow_prices = np.where(np.random.rand(len(df_go_live)) < 0.7, 0, np.random.exponential(scale=30, size=len(df_go_live)))
    shadow_prices = np.clip(shadow_prices, 0, 180)  # Limit the maximum value to 180
    df_go_live['shadow_price'] = shadow_prices

    df_go_live['Flow_FB'] = np.random.randint(-3200, 3200, df_go_live.shape[0])
    
    ##################################################################################################################################
    
    return df_go_live


#######################################################################################################################################
                                            


# Function to get day-ahead prices for a given zone
def fetch_data_for_zone(zone_code, start_date, end_date):
    from_utc = start_date.strftime("%Y%m%d%H%M")  # YYYYMMDDHHMM format
    to_utc = end_date.strftime("%Y%m%d%H%M")
    
    params = {
        "documentType": "A44",  # Day-ahead prices document type
        "in_Domain": zone_code,
        "out_Domain": zone_code,
        "periodStart": from_utc,
        "periodEnd": to_utc,
        "securityToken": "394c6def-8205-4ab5-9838-561a518d97a5"
    }
    
    response = requests.get("https://web-api.tp.entsoe.eu/api", params=params)
    
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Error fetching data: {response.status_code}, {response.text}")

# Function to parse XML response into a DataFrame
def parse_price_data(xml_data, zone_name):
    root = etree.fromstring(xml_data)
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
    
    # List to hold extracted data
    data = []

    # Loop through each time series in the response
    for time_series in root.findall(".//ns:TimeSeries", namespaces=ns):
        # Extract relevant periods
        for period in time_series.findall(".//ns:Period", namespaces=ns):
            start_time_str = period.find(".//ns:start", namespaces=ns).text
            start_time = pd.to_datetime(start_time_str)

            # Determine resolution
            resolution = period.find(".//ns:resolution", namespaces=ns).text
            if resolution != "PT60M":  # Skip if not 60-minute resolution
                continue
            time_step = timedelta(hours=1)

            # Loop through the points (prices)
            for point in period.findall(".//ns:Point", namespaces=ns):
                position = int(point.find(".//ns:position", namespaces=ns).text) - 1
                price = point.find(".//ns:price.amount", namespaces=ns).text

                # Handle non-numeric values
                try:
                    price = float(price)
                except ValueError:
                    price = None  # Set to None or NaN if not numeric

                # Calculate the actual datetime for each price point
                point_time = start_time + position * time_step
                data.append([zone_name, point_time, price])
    
    # Convert list to DataFrame
    df1 = pd.DataFrame(data, columns=["Zone", "Datetime", "Price"])
    df1['Price'] = pd.to_numeric(df1['Price'], errors='coerce')
    df1.dropna(subset=['Price'], inplace=True)
    df1['Datetime'] = df1['Datetime'] + pd.Timedelta(hours=2)

    return df1

# Function to fetch data for Germany
def fetch_germany_data(start_date, end_date):
    germany_zone = "10Y1001A1001A82H"  # Germany with 15-minute resolution
    xml_data = fetch_data_for_zone(germany_zone, start_date, end_date)
    return parse_price_data(xml_data, "DE")

# Function to fetch data for all other bidding zones
def fetch_other_zones_data(start_date, end_date):
    bidding_zones = {
        "DK1": "10YDK-1--------W",
        "DK2": "10YDK-2--------M",
        "NO1": "10YNO-1--------2",
        "NO2": "10YNO-2--------T",
        "NO3": "10YNO-3--------J",
        "NO4": "10YNO-4--------9",
        "NO5": "10Y1001A1001A48H",
        "SE1": "10Y1001A1001A44P",
        "SE2": "10Y1001A1001A45N",
        "SE3": "10Y1001A1001A46L",
        "SE4": "10Y1001A1001A47J",
        "FI": "10YFI-1--------U",
        "ES": "10Y1001A1001A39I",
        "PL": "10YPL-AREA-----S",
        "NL": "10YNL----------L",
        "LI": "10YLT-1001A0008Q"
    }

#


    dfs = []
    for zone_name, zone_code in bidding_zones.items():
        xml_data = fetch_data_for_zone(zone_code, start_date, end_date)
        df2 = parse_price_data(xml_data, zone_name)
        dfs.append(df2)
    
    # Combine all non-Germany data into one DataFrame
    return pd.concat(dfs, ignore_index=True)

# Function to fetch both Germany and other zones data
def fetch_data_prices(start_date, end_date):
    # Fetch data for Germany
    germany_df = fetch_germany_data(start_date, end_date)
    
    # Fetch data for all other zones
    other_zones_df = fetch_other_zones_data(start_date, end_date)
    
    return germany_df, other_zones_df









# Split the dataframe into Border, NetPosition, and Remaining DataFrames
def split_dataframes(df):
    # Split into Border_df and Net_df based on 'cnecName'
    Border_df = df[df['cnecName'].str.contains("Border", case=False, na=False)].copy()
    
    
    Net_df = df[df['cnecName'].str.contains("NetPosition", case=False, na=False)].copy()
    
    
    
    # Add 'flow' column to Border_df
    Border_df['flow'] = Border_df['biddingZoneFrom'] + "->" + Border_df['biddingZoneTo']
    

    # Remaining data excluding Border_df and Net_df
    Remaining_df = df[~df.index.isin(Border_df.index) & ~df.index.isin(Net_df.index)]
    
    

    # Store the dataframes in session state
    st.session_state['Border_df'] = Border_df
    st.session_state['Net_df'] = Net_df
    st.session_state['Remaining_df'] = Remaining_df
    

def main_page():
    st.title("API Data Fetcher")

    # Date input
    start_date = st.date_input("Start Date", value=datetime(2024, 10, 1))
    end_date = st.date_input("End Date", value=datetime(2024, 10, 10))

    # Calculate the date difference in days
    date_difference = (end_date - start_date).days

    if date_difference > 31:
        st.warning("JAO API cannot manage date ranges longer than one month. Please try again with a shorter range.")
    else:
        if st.button("Fetch Data"):
            
            
            
            
            start_datetime = datetime.combine(start_date, datetime.min.time())  # Add time to start_date
            start_date_minus_one_hour = start_datetime - timedelta(hours=10)  # Subtract one hour
           
            df = fetch_data(start_date_minus_one_hour, end_date)

            germany_df, other_zones_df = fetch_data_prices(start_date_minus_one_hour, end_date)
            st.session_state['germany_df'] = germany_df
            st.session_state['other_zones_df'] = other_zones_df

            Net_df = df[df['cnecName'].str.contains("NetPosition", case=False, na=False)].copy()

            if not df.empty:
                # Split and store DataFrames in session state
                split_dataframes(df)
                st.success("Data fetched and DataFrames created!")
                st.dataframe(df.head())  # Preview first few rows
            else:
                st.warning("No data returned from the API.")

def plot_page():
    st.title("Plot Page")

    # Check if DataFrames are available in session state
    if 'Remaining_df' in st.session_state and 'Border_df' in st.session_state and 'Net_df' in st.session_state:
        Remaining_df = st.session_state['Remaining_df']
        Border_df = st.session_state['Border_df']
        Net_df = st.session_state['Net_df']

        # Create a display list for TSOs, replacing '' with 'AC_CNEC'
        display_tso_options = ['AC_CNEC' if tso == '' else tso for tso in Remaining_df['tso'].unique()]

        # Callback function to update TSO selection and reset CNEC selection
        def update_tso():
            st.session_state['tso_selected'] = st.session_state['tso_selectbox']
            # Reset CNEC selection when TSO is changed
            st.session_state.pop('selected_cnec', None)

        # Callback function to update CNEC selection
        def update_cnec():
            st.session_state['selected_cnec'] = st.session_state['cnec_selectbox']

        # Callback function to update selected columns for Remaining_df
        def update_columns_remaining():
            st.session_state['columns_to_plot_remaining'] = st.session_state['columns_to_plot_remaining_select']

        # Callback function to update selected columns for Border_df
        def update_columns_border():
            st.session_state['columns_to_plot_border'] = st.session_state['columns_to_plot_border_select']

        # Callback function to update selected columns for Net_df
        def update_columns_net():
            st.session_state['columns_to_plot_net'] = st.session_state['columns_to_plot_net_select']

        #### Save TSO selection in session_state ####
        if 'tso_selected' not in st.session_state:
            st.session_state['tso_selected'] = display_tso_options[0]  # Default selection

        # Select a TSO from Remaining_df
        st.selectbox(
            "Select a TSO",
            display_tso_options,
            index=display_tso_options.index(st.session_state['tso_selected']),
            key='tso_selectbox',
            on_change=update_tso  # Use callback to update session state and reset CNEC
        )

        # Map 'AC_CNEC' back to '' when filtering Remaining_df
        selected_tso = '' if st.session_state['tso_selected'] == 'AC_CNEC' else st.session_state['tso_selected']
        
        # Filter the Remaining_df based on the selected TSO
        tso_filtered_df = Remaining_df[Remaining_df['tso'] == selected_tso]

        #### Save CNEC selection in session_state ####
        if tso_filtered_df['tso'].iloc[0] == "ENERGINET":
            tso_filtered_df['cnec_cont'] = tso_filtered_df['cnecName'] + " + " + tso_filtered_df['contName']
            if 'selected_cnec' not in st.session_state:
                st.session_state['selected_cnec'] = tso_filtered_df['cnec_cont'].unique()[0]  # Default selection for new TSO
            st.selectbox(
                "Select a CNEC",
                tso_filtered_df['cnec_cont'].unique(),
                index=list(tso_filtered_df['cnec_cont'].unique()).index(st.session_state['selected_cnec']),
                key='cnec_selectbox',
                on_change=update_cnec  # Use callback to update session state
            )
            filtered_remaining_df = tso_filtered_df[tso_filtered_df['cnec_cont'] == st.session_state['selected_cnec']]
        else:
            if 'selected_cnec' not in st.session_state:
                st.session_state['selected_cnec'] = tso_filtered_df['cnecName'].unique()[0]  # Default selection for new TSO
            st.selectbox(
                "Select a CNEC",
                tso_filtered_df['cnecName'].unique(),
                index=list(tso_filtered_df['cnecName'].unique()).index(st.session_state['selected_cnec']),
                key='cnec_selectbox',
                on_change=update_cnec  # Use callback to update session state
            )
            filtered_remaining_df = tso_filtered_df[tso_filtered_df['cnecName'] == st.session_state['selected_cnec']]

        #### Save columns to plot for Remaining_df in session_state ####
        if 'columns_to_plot_remaining' not in st.session_state:
            st.session_state['columns_to_plot_remaining'] = ['ram']  # Default columns

        # Choose columns to plot for Remaining_df
        st.multiselect(
            "Select columns to plot for Remaining_df",
            options=['ram', 'minFlow','maxFlow', 'u', 'imax', 'fmax', 'frm', 'fref', 'fall', 'fnrao', 'amr', 'aac', 'iva', 'shadow_price', 'Flow_FB'],
            default=st.session_state['columns_to_plot_remaining'],
            key='columns_to_plot_remaining_select',
            on_change=update_columns_remaining  # Use callback to update session state
        )

        # Plot the selected CNEC for the Remaining_df
        if st.session_state['columns_to_plot_remaining']:
            fig = px.line(filtered_remaining_df, x='dateTimeCET', y=st.session_state['columns_to_plot_remaining'], title=f"Plot for {st.session_state['selected_cnec']} (Remaining_df)")
            st.plotly_chart(fig)

        #### Plot for Border_df ####
        st.subheader("Plot CNECs for Border")

        # Save Border_df CNEC selection in session_state
        if 'selected_flow_border' not in st.session_state:
            st.session_state['selected_flow_border'] = Border_df['cnecName'].unique()[0]  # Default selection

        st.selectbox(
            "Select a CNEC from Border_df",
            Border_df['cnecName'].unique(),
            index=list(Border_df['cnecName'].unique()).index(st.session_state['selected_flow_border']),
            key='border_cnec_selectbox',
            on_change=lambda: st.session_state.update({'selected_flow_border': st.session_state['border_cnec_selectbox']})
        )

        filtered_border_df = Border_df[Border_df['cnecName'] == st.session_state['selected_flow_border']]

        # Save columns to plot for Border_df in session_state
        if 'columns_to_plot_border' not in st.session_state:
            st.session_state['columns_to_plot_border'] = ['ram']  # Default columns for Border_df

        # Choose columns to plot for Border_df
        st.multiselect(
            "Select columns to plot for Border_df",
            options=['ram', 'minFlow','maxFlow', 'u', 'imax', 'fmax', 'frm', 'fref', 'fall', 'fnrao', 'amr', 'aac', 'iva', 'shadow_price', 'Flow_FB'],
            default=st.session_state['columns_to_plot_border'],
            key='columns_to_plot_border_select',
            on_change=update_columns_border  # Use callback to update session state
        )

        # Plot the selected CNEC for the Border_df
        if st.session_state['columns_to_plot_border']:
            fig_border = px.line(filtered_border_df, x='dateTimeCET', y=st.session_state['columns_to_plot_border'], title=f"Plot for {st.session_state['selected_flow_border']} (Border_df)")
            st.plotly_chart(fig_border)

        #### Plot for Net_df ####
        st.subheader("Plot CNECs for Net Position")

        # Save Net_df CNEC selection in session_state
        if 'selected_cnec_net' not in st.session_state:
            st.session_state['selected_cnec_net'] = Net_df['cnecName'].unique()[0]  # Default selection

        st.selectbox(
            "Select a CNEC from Net_df",
            Net_df['cnecName'].unique(),
            index=list(Net_df['cnecName'].unique()).index(st.session_state['selected_cnec_net']),
            key='net_cnec_selectbox',
            on_change=lambda: st.session_state.update({'selected_cnec_net': st.session_state['net_cnec_selectbox']})
        )

        filtered_net_df = Net_df[Net_df['cnecName'] == st.session_state['selected_cnec_net']]

        # Save columns to plot for Net_df in session_state
        if 'columns_to_plot_net' not in st.session_state:
            st.session_state['columns_to_plot_net'] = ['ram']  # Default columns for Net_df

        # Choose columns to plot for Net_df
        st.multiselect(
            "Select columns to plot for Net_df",
            options=['ram', 'minFlow','maxFlow', 'fmax', 'shadow_price', 'Flow_FB'],
            default=st.session_state['columns_to_plot_net'],
            key='columns_to_plot_net_select',
            on_change=update_columns_net  # Use callback to update session state
        )

        # Plot the selected CNEC for the Net_df
        if st.session_state['columns_to_plot_net']:
            fig_net = px.line(filtered_net_df, x='dateTimeCET', y=st.session_state['columns_to_plot_net'], title=f"Plot for {st.session_state['selected_cnec_net']} (Net_df)")
            st.plotly_chart(fig_net)

    else:
        st.warning("No DataFrame found. Please go to the Main Page and initiate the API call.")



def map_page():
    
    ##########################################################################################
                                ### MAP###
    ###########################################################################################
    st.title("Hourly summary - Map")

    # Date selection
    Remaining_df = st.session_state.get('Remaining_df', None)
    
    
    Net_df = st.session_state.get('Net_df', None)

    
    def update_hour():
        st.session_state['selected_hour'] = st.session_state['hour_selectbox']
        
    def update_date():
        st.session_state['selected_date'] = st.session_state['date_selectbox']
        

    if Remaining_df is not None:
    # Extract unique dates from Remaining_df['dateTimeUtc'] and convert them to datetime.date
        Remaining_df['date'] = pd.to_datetime(Remaining_df['dateTimeUtc']).dt.date
        unique_dates = Remaining_df['date'].unique()
        unique_dates = unique_dates.tolist()  # Convert unique_dates to a list
        
        
        if 'selected_date' not in st.session_state:
            st.session_state['selected_date'] = unique_dates[0]  # Default to the first date
        

        
        
    # Date picker with only available dates from Remaining_df
        selected_date = st.selectbox(
            "Select a date",
            options=unique_dates,
            index=unique_dates.index(st.session_state['selected_date']),
            key='date_selectbox',
            on_change=update_date  # Callback to update session state when date changes
)
        
        
        
        filtered_df_by_date = Remaining_df[Remaining_df['date'] == selected_date]

    # Extract unique hours for the selected date (formatted as "HH:MM")
        filtered_df_by_date['hour'] = pd.to_datetime(filtered_df_by_date['dateTimeUtc']).dt.strftime('%H:%M')
        unique_hours = filtered_df_by_date['hour'].unique()
        unique_hours = unique_hours.tolist()  # Convert unique_hours to a list
        
        
        if 'selected_hour' not in st.session_state or st.session_state['selected_hour'] not in unique_hours:
            st.session_state['selected_hour'] = unique_hours[0]  # Default to the first date

    # Hour picker with only available hours for the selected date
        selected_hour = st.selectbox(
            "Select an hour",
            options=unique_hours,
            index=unique_hours.index(st.session_state['selected_hour']),
            key='hour_selectbox',
            on_change=update_hour  # Callback to update session state when hour changes
)
        
        

    # Combine the selected date and hour into a single datetime object
        selected_datetime = datetime.combine(selected_date, datetime.strptime(selected_hour, "%H:%M").time())

    # Format the datetime to the desired string format with +00:00
        formatted_datetime = selected_datetime.strftime("%Y-%m-%d %H:%M:%S") + "+00:00"
        st.write(f"Visualizing data for datetime: {formatted_datetime}")

    # Filter Remaining_df based on the selected date and hour
        tso_display_mapping = {'ENERGINET': 'ENERGINET', 'STATNETT': 'STATNETT', 'FINGRID': 'FINGRID', 'SVK': 'SVK', 'AC_CNEC': ''}

# Initialize the TSO selection in session state if not already present
        if 'selected_tso' not in st.session_state:
            st.session_state['selected_tso'] = ['ENERGINET', 'STATNETT', 'FINGRID', 'SVK', 'AC_CNEC']

# Multiselect with display names from the mapping
        selected_display_tso = st.multiselect(
            "Select TSO to table",
            options=list(tso_display_mapping.keys()),  # Show 'AC_CNEC' as an empty string
            format_func=lambda x: tso_display_mapping[x] if tso_display_mapping[x] != '' else 'AC_CNEC',  # Show 'AC_CNEC' properly in dropdown
            default=st.session_state['selected_tso']
)

# Update session state with the new selection
        st.session_state['selected_tso'] = selected_display_tso

# Filter the DataFrame using the actual TSO values
        tso_filtered_df = Remaining_df[Remaining_df['tso'].isin(selected_display_tso)]

# Display or use tso_filtered_df as needed
       





    
    
        tso_filtered_df = tso_filtered_df[tso_filtered_df['dateTimeCET'] == formatted_datetime]
        
        
        Net_df_filtered = Net_df[Net_df['dateTimeCET'] == formatted_datetime]


    ## Filter for rows where tso == 'ENERGINET'
        tso_filtered_df_energinet = tso_filtered_df[tso_filtered_df['tso'] == 'ENERGINET'].copy()

# Overwrite the 'cnecName' column by concatenating 'cnecName' and 'contName'
        tso_filtered_df_energinet['cnecName'] = tso_filtered_df_energinet['cnecName'] + " + " + tso_filtered_df_energinet['contName']

# Now update the original DataFrame only for those rows where tso == 'ENERGINET'
        tso_filtered_df.loc[tso_filtered_df['tso'] == 'ENERGINET', 'cnecName'] = tso_filtered_df_energinet['cnecName']









    # Coordinates and flow data (You can modify these values based on your data)
    COORDINATES = {
        'DK1': [56.108059, 9.199232],
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
        'NL':  [52.279871, 4.805468],
        'DE':  [50.101574, 8.614998],
        'PL':  [51.918670, 19.553589],
        'LI':  [55.379634, 24.025465],
        'ES':  [58.861672, 25.806059],
        
        
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
        'NL':  40,
        'DE':  70,
        'PL':  14,
        'LI':  30,
        'ES':  25,
    }

    net_price = {
        'DK1': 100,   # Example net position
        'DK2': 50,
        'SE4': 30,
        'SE3': 80,
        'SE2': 20,
        'SE1': 10,
        'NO4': 40,
        'NO3': 50,
        'NO1': 30,
        'NO2': 25,
        'NO5': 10,
        'FI':  60,    # Example net position for Finland
        'NL':  40,
        'DE':  70,
        'PL':  14,
        'LI':  30,
        'ES':  25,
    }

    #########################################################################################################

                                                ###BORDER CNECS###
    #########################################################################################################
    
    Border_df = st.session_state.get('Border_df', None)
    Border_df = Border_df[Border_df['dateTimeCET'] == formatted_datetime]
    
    
    germany_df = st.session_state['germany_df']
    germany_df = germany_df[germany_df['Datetime'] == formatted_datetime]
    
    other_zones_df = st.session_state['other_zones_df']

    other_zones_df = other_zones_df[other_zones_df['Datetime'] == formatted_datetime]
    germany_df['Price'] = germany_df['Price'].round(0).astype(int)
    other_zones_df['Price'] = other_zones_df['Price'].round(0).astype(int)

    net_price = pd.concat([germany_df, other_zones_df], ignore_index=True)
    
    net_price.rename(columns={'Zone': 'region'}, inplace=True)
    net_price = net_price.set_index('region')['Price'].to_dict()
    
    all_zones = ['DK1', 'DK2', 'SE4', 'SE3', 'SE2', 'SE1', 'NO4', 'NO3', 'NO1', 'NO2', 'NO5', 'FI', 'NL', 'DE', 'PL', 'LI', 'ES']

    for zone in all_zones:
        if zone not in net_price:
            net_price[zone] = np.nan  # Assign NaN if price is missing for the zone
    
    
    
    ##### DK1
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_SB', 'DK2')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_SB', 'DK2')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_DE', 'DE')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_DE', 'DE')

    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_CO', 'NL')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_CO', 'NL')

    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_SK', 'NO2')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_SK', 'NO2')
    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_KS', 'SE3')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_KS', 'SE3')

    
    
    ##### DK2
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK2_SB', 'DK1')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK2_SB', 'DK1')    
    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK2_KO', 'DE')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK2_KO', 'DE')
    
    ##### SE4
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE4_BC', 'DE')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE4_BC', 'DE')

    
    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE4_NB', 'PL')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE4_NB', 'PL')

    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE4_SP', 'LI')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE4_SP', 'LI')

    
    
    ### South west link
    
    
    
    
    
    ####  SE3
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE3_KS', 'DK1')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE3_KS', 'DK1')


    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE3_FS', 'FI')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE3_FS', 'FI')

    ####  FI
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('FI_FS', 'SE3')    
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('FI_FS', 'SE3')    

 
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('FI_EL', 'ES')   
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('FI_EL', 'ES')    


    #### NO2
    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('NO2_SK', 'DK1')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('NO2_SK', 'DK1')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('NO2_ND', 'NL')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('NO2_ND', 'NL')

    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('NO2_NK', 'DE')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('NO2_NK', 'DE')

    Border_df['flow'] =  Border_df['biddingZoneFrom']+'-'+Border_df['biddingZoneTo']
    
    
    
    
    
    
    def filter_for_zones(df, zone1, zone2, column):
    # Create the forward and reverse search patterns
        forward = f"{zone1}-{zone2}"
        reverse = f"{zone2}-{zone1}"
    
    # Filter the DataFrame for either forward or reverse pattern
        filtered_df = df[(df[column] == forward) | (df[column] == reverse)]
        if not filtered_df.empty:
            max_flow_row = filtered_df.loc[filtered_df['Flow_FB'].abs().idxmax()]
            return max_flow_row
        else:
            return None

    max_flow_row = filter_for_zones(Border_df, 'DK1', 'DK2', 'flow')

    
    
    

    
     
    
    
    
    
    
    
    ##################################################################################################
                                        ## FLOW UPDATE ##
    ##################################################################################################
    # INITIAL VALUES
    
    flows = {
        ('DK1', 'DK2'): 0,
        ('DK1', 'SE3'): 0,
        ('DK1', 'NO2'): 0,
        ('DK2', 'SE4'): 0,
        ('SE4', 'SE3'): 0,
        ('SE3', 'NO1'): 0,
        ('SE3', 'SE2'): 0,
        ('NO1', 'NO2'): 0,
        ('NO1', 'NO3'): 0,
        ('NO1', 'NO5'): 0,
        ('NO2', 'NO5'): 0,
        ('NO5', 'NO3'): 0,
        ('NO3', 'SE2'): 0,
        ('NO3', 'NO4'): 0,
        ('NO4', 'SE2'): 0,  
        ('SE1', 'NO4'): 0,  
        ('SE1', 'SE2'): 0,  
        ('SE1', 'FI'):  0,  
        ('FI', 'SE3'):  0,  
        ('FI', 'NO4'):  0,
        ('NO2', 'NL'):  0,
        ('DK1', 'NL'):  0,
        ('NO2', 'DE'):  0, 
        ('DK1', 'DE'):  0, 
        ('DK2', 'DE'):  0, 
        ('SE4', 'PL'):  0, 
        ('SE4', 'DE'):  0,
        ('SE4', 'LI'):  0, 
        ('FI',  'ES'):  0,
    }
    flows_copy = flows.copy()

# Iterate over the original flows dictionary
    for (zone1, zone2), value in list(flows.items()):  # Use list(flows.items()) to avoid modifying the dictionary during iteration
    # Filter for the corresponding zones and get the row with the maximum flow
        max_flow_row = filter_for_zones(Border_df, zone1, zone2, 'flow')
    
    # If a valid row is returned, update the flows dictionary
        if max_flow_row is not None and not max_flow_row.empty:
            flows[(zone1, zone2)] = max_flow_row['Flow_FB']  # Overwrite the original flows directly

        else:
        # Optional: log if no valid row is found
            st.write(f"No valid flow found for {zone1}->{zone2}")



    arrow_scale = 3
    fig = go.Figure()





    def add_arrow(fig, source, target, flow_value, arrow_scale):
        source_coord = np.array(COORDINATES[source])
        target_coord = np.array(COORDINATES[target])
        
        
        
        if flow_value < 0:

            source_coord, target_coord = target_coord, source_coord
            source, target = target, source
            flow_value = abs(flow_value)
        
        line_width = flow_value * arrow_scale / max(flows.values())+1
        
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
            text=[f"{flow_value:.1f}"],
            textfont=dict(size=14, color='blue'),
            textposition="middle center",
            hoverinfo='text',
            hovertext=f"{source} -> {target}: {flow_value:.1f}"
        ))
        
    
    def add_curved_arrow(fig, source, target, flow_value, arrow_scale):



        source_coord = np.array(COORDINATES[source])
        target_coord = np.array(COORDINATES[target])
        
        if flow_value < 0:
            source_coord, target_coord = target_coord, source_coord
            source, target = target, source
            flow_value = abs(flow_value)        
            # Always calculate the intermediate points using the original source and target coordinates
        mid_point_lat = (source_coord[0] + target_coord[0]) / 2
        mid_point_lon = (source_coord[1] + target_coord[1]) / 2

    # Add curvature by adjusting the mid-point and adding control points
        control_point_1_lat = mid_point_lat +1  # Adjust for more curvature if needed
        control_point_1_lon = mid_point_lon + 1

        control_point_2_lat = mid_point_lat +1  # Adjust for more curvature if needed
        control_point_2_lon = mid_point_lon + 1
        line_width = flow_value * arrow_scale / max(flows.values()) + 1

            # Add a curved line by adding intermediate points (manual control points)
        fig.add_trace(go.Scattermapbox(
            lon=[source_coord[1], control_point_1_lon, control_point_2_lon, target_coord[1]],
            lat=[source_coord[0], control_point_1_lat, control_point_2_lat, target_coord[0]],
            mode='lines',
            line=dict(width=line_width, color='blue'),  # Adjust line color if needed
            opacity=0.7,
            hoverinfo='none'
    ))

    # If flow is negative, reverse source and target
        
        



    # Add a curved line by adding intermediate points (manual control points)


    # Add an arrowhead at the target location
        v = target_coord - np.array([control_point_1_lat, control_point_1_lon])
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
            fillcolor='blue',  # Adjust arrowhead color if needed
            line_color='blue',
            hoverinfo='none'
    ))

    # Add the flow value text in the middle of the curved arrow
        fig.add_trace(go.Scattermapbox(
            lon=[mid_point_lon+1],
            lat=[mid_point_lat+1],
            mode='text',
            text=[f"{flow_value:.1f}"],
            textfont=dict(size=14, color='blue'),
            textposition="middle center",
            hoverinfo='text',
            hovertext=f"{source}_SWL -> {target}_SWL: {flow_value:.1f}"
    ))


# Iterate through flows where keys are tuples (source, target)
    for (source, target), flow_value in flows.items():
    # Add the arrow using the parsed source and target  
             add_arrow(fig, source, target, flow_value, arrow_scale)
             
             
    new_df = Border_df[Border_df['cnecName'].str.contains('SWL', case=False, na=False)]         
    #new_df['biddingZoneTo'] = new_df['biddingZoneTo'].replace('46Y000000000007M', 'SE4')
    #new_df['biddingZoneFrom'] = new_df['biddingZoneFrom'].replace('46Y000000000007M', 'SE4')
    
    new_df['biddingZoneTo'] = new_df['biddingZoneTo'].replace('SE4_SWL', 'SE3')
    new_df['biddingZoneFrom'] = new_df['biddingZoneFrom'].replace('SE4_SWL', 'SE3')
    
    
    #new_df['biddingZoneFrom'] = new_df['biddingZoneFrom'].replace('46Y000000000008K', 'SE3')
    #new_df['biddingZoneTo'] = new_df['biddingZoneTo'].replace('46Y000000000008K', 'SE3')  
      
    new_df['biddingZoneFrom'] = new_df['biddingZoneFrom'].replace('SE3_SWL', 'SE4')
    new_df['biddingZoneTo'] = new_df['biddingZoneTo'].replace('SE3_SWL', 'SE4')    
    
    new_df['flow'] =  new_df['biddingZoneFrom']+'-'+new_df['biddingZoneTo']
    
    
    max_flow_row = filter_for_zones(new_df, 'SE3', 'SE4', 'flow')   
 
    source = max_flow_row['biddingZoneFrom']
    target = max_flow_row['biddingZoneTo']
    flow_value = max_flow_row['Flow_FB']

# Call the add_curved_arrow function with the extracted values
    add_curved_arrow(fig=fig, source=source, target=target, flow_value=flow_value, arrow_scale=0.5)             
             
             
             
             
             
             
             
             
             
             
             
             
             

# Add the region markers and their values
    for region, coord in COORDINATES.items():
        max_value = max([v for v in net_price.values() if pd.notna(v)])
        min_value = min([v for v in net_price.values() if pd.notna(v)])
        if pd.isna(net_price[region]):
            color1 = 'black'
        else: 
            normalized_value = normalize(net_price[region], max_value, min_value)
            color1 = f'rgba({255 * (1 - normalized_value)}, 0, 0, {1 - normalized_value + 0.3})'

    # Add region markers first (these will be under the text)
        fig.add_trace(go.Scattermapbox(
            lon=[coord[1]],  # Longitude
            lat=[coord[0]],  # Latitude
            mode='markers',  # Markers only
            marker=dict(
            size=50,  # Set marker size
            color=color1  # Dynamic color based on value
        ),
    ))

    # Add the text inside the markers (always rendered last, on top)
        fig.add_trace(go.Scattermapbox(
            lon=[coord[1]],
            lat=[coord[0]],
            mode='text',
            text=[net_price[region]],
            textfont=dict(size=24, color='white'),
            textposition="middle center",
            hoverinfo='text',
            hovertext=f"{region}, {net_price[region]}"
        ))
# Update the map layout
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                zoom=4,
                center=dict(lat=61, lon=15),
        ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            width=1200,
            height=800,
            showlegend=False
    )

# Display the figure in Streamlit
    st.plotly_chart(fig)
    st.subheader("TSO - PTDFs")

    tso_filtered_df_no_index = tso_filtered_df[['cnecName','tso','Flow_FB','ram','shadow_price','ptdf_DK1','ptdf_DK1_CO','ptdf_DK1_DE','ptdf_DK1_KS','ptdf_DK1_SK','ptdf_DK1_SB','ptdf_DK2','ptdf_DK2_KO','ptdf_DK2_SB','ptdf_FI','ptdf_FI_EL','ptdf_FI_FS','ptdf_NO1','ptdf_NO2','ptdf_NO2_ND','ptdf_NO2_SK','ptdf_NO2_NK','ptdf_NO3','ptdf_NO4','ptdf_NO5','ptdf_SE1','ptdf_SE2','ptdf_SE3','ptdf_SE3_FS','ptdf_SE3_KS','ptdf_SE3_SWL','ptdf_SE4','ptdf_SE4_BC','ptdf_SE4_NB','ptdf_SE4_SP','ptdf_SE4_SWL']].reset_index(drop=True)
    
    
    tso_filtered_df_no_index = tso_filtered_df_no_index.sort_values(by='shadow_price', ascending=False)

         
    st.dataframe(tso_filtered_df_no_index.set_index('cnecName'))
    st.subheader("NetPosition")

    columns_to_display = ['Flow_FB', 'cnecName']

# Ensure that the selected columns exist in the filtered DataFrame
    if all(col in Net_df_filtered.columns for col in columns_to_display):
        st.dataframe(Net_df_filtered[columns_to_display].set_index('cnecName'))
    else:
        st.write("Some of the required columns are missing from the DataFrame.")
    
    
#########################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Display the filtered dataframe
   # if not tso_filtered_df.empty:
   #     remaining_table = tso_filtered_df[['cnec_cont', 'ptdf_DK1','ptdf_DK1_CO','ptdf_DK1_DE','ptdf_DK1_KS','ptdf_DK1_SK','ptdf_DK1_SB','ptdf_DK2','ptdf_DK2_KO','ptdf_DK2_SB','ptdf_FI','ptdf_FI_EL','ptdf_FI_FS','ptdf_NO1','ptdf_NO2','ptdf_NO2_ND','ptdf_NO2_SK','ptdf_NO2_NK','ptdf_NO3','ptdf_NO4','ptdf_NO5','ptdf_SE1','ptdf_SE2','ptdf_SE3','ptdf_SE3_FS','ptdf_SE3_KS','ptdf_SE3_SWL','ptdf_SE4','ptdf_SE4_BC','ptdf_SE4_NB','ptdf_SE4_SP','ptdf_SE4_SWL','ram','']]  # Customize with specific columns like 'ptdf_DK1'
   #     st.dataframe(remaining_table.set_index('cnec_cont'))
   # else:
   #     st.write("No data available for the selected datetime.")

  




    ###############################################################################################
                                        ###END###
    ###############################################################################################


# Function for the flow plot page
# Function for the flow plot page


def flow_page():
    # Define the available flows in the desired format
    
    def update_selected_flows():
        st.session_state['selected_flows'] = st.session_state['selected_flows_select']
    
    available_flows = [
        'DK1->DK2', 'DK1->SE3', 'DK1->NO2', 'DK2->SE4',
        'SE4->SE3', 'SE3->NO1', 'SE3->SE2', 'NO1->NO2',
        'NO1->NO3', 'NO1->NO5', 'NO2->NO5', 'NO5->NO3',
        'NO3->SE2', 'NO3->NO4', 'NO4->SE2', 'SE1->NO4',
        'SE1->SE2', 'SE1->FI', 'FI->SE3', 'FI->NO4',
        'NO2->NL', 'DK1->NL', 'NO2->DE', 'DK1->DE',
        'DK2->DE', 'SE4->PL', 'SE4->DE', 'SE4->LI',
        'FI->ES'
    ]
    #########################################################################################################
    ### BORDER CNECS ###
    #########################################################################################################
    Border_df = st.session_state.get('Border_df', None)

    ##### DK1
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_SB', 'DK2')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_SB', 'DK2')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_DE', 'DE')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_DE', 'DE')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_CO', 'NL')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_CO', 'NL')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_SK', 'NO2')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_SK', 'NO2')
    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK1_KS', 'SE3')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK1_KS', 'SE3')

    ##### DK2
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK2_SB', 'DK1')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK2_SB', 'DK1')    
    
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('DK2_KO', 'DE')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('DK2_KO', 'DE')
    
    ##### SE4
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE4_BC', 'DE')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE4_BC', 'DE')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE4_NB', 'PL')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE4_NB', 'PL')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE4_SP', 'LI')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE4_SP', 'LI')

    #### SE3
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE3_KS', 'DK1')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE3_KS', 'DK1')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('SE3_FS', 'FI')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('SE3_FS', 'FI')

    #### FI
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('FI_FS', 'SE3')    
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('FI_FS', 'SE3')    

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('FI_EL', 'ES')   
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('FI_EL', 'ES')    

    #### NO2
    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('NO2_SK', 'DK1')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('NO2_SK', 'DK1')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('NO2_ND', 'NL')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('NO2_ND', 'NL')

    Border_df['biddingZoneTo'] = Border_df['biddingZoneTo'].replace('NO2_NK', 'DE')
    Border_df['biddingZoneFrom'] = Border_df['biddingZoneFrom'].replace('NO2_NK', 'DE')

    # Create the flow column combining biddingZoneFrom and biddingZoneTo
    Border_df['flow'] =  Border_df['biddingZoneFrom'] + '-' + Border_df['biddingZoneTo']
    
    #########################################################################################################
    ### FUNCTION TO FILTER MAXIMUM FLOW ###
    #########################################################################################################

    def filter_for_zones(df, zone1, zone2, column):
        
        
        
        
        
        # Create the forward and reverse search patterns
        forward = f"{zone1}-{zone2}"
        reverse = f"{zone2}-{zone1}"
    
        # Filter the DataFrame for either forward or reverse pattern
        filtered_df = df[(df[column] == forward) | (df[column] == reverse)]
        if not filtered_df.empty:
            max_flow_row = filtered_df.loc[filtered_df['Flow_FB'].abs().idxmax()]
            return max_flow_row
        else:
            return None
    def get_max_flow_per_time_normalized(df, zone1, zone2, column):

    # Create the forward and reverse search patterns
        forward = f"{zone1}-{zone2}"
        reverse = f"{zone2}-{zone1}"
    
    # Filter the DataFrame for either forward or reverse pattern
        filtered_df = df[(df[column] == forward) | (df[column] == reverse)]
    
        if filtered_df.empty:
            return pd.DataFrame(columns=['dateTimeCET', 'maxFlow_fb', 'flow_direction'])  # Return empty DataFrame if no matches

    # Group by dateTimeCET and find the row with the maximum Flow_FB for each time point
        max_flow_per_time = filtered_df.groupby('dateTimeCET').apply(lambda x: x.loc[x['Flow_FB'].abs().idxmax()])

    # Create a new DataFrame with just dateTimeCET and maxFlow
        result_df = max_flow_per_time[['dateTimeCET', 'Flow_FB','maxFlow','minFlow', column]].copy()

    # Normalize the flow direction
        result_df['maxFlow_fb'] = result_df.apply(
            lambda row: row['Flow_FB'] if row[column] == forward else -row['Flow_FB'], axis=1
    )

    # Ensure the flow direction is always zone1 -> zone2
        result_df['flow_direction'] = forward

    # Drop the original Flow_FB and flow columns (keeping only dateTimeCET, maxFlow, flow_direction)
        result_df = result_df[['dateTimeCET','maxFlow_fb','maxFlow','minFlow', 'flow_direction']]

        return result_df
    
    
    
    # Iterate over the flows and update with maximum values
    flows = {('DK1', 'DK2'): 0, ('DK1', 'SE3'): 0, ('DK1', 'NO2'): 0, ('DK2', 'SE4'): 0}
    for (zone1, zone2), value in list(flows.items()):
        max_flow_row = filter_for_zones(Border_df, zone1, zone2, 'flow')
        if max_flow_row is not None:
            flows[(zone1, zone2)] = max_flow_row['Flow_FB']

    #########################################################################################################
    ### FLOW SELECTION AND PLOTTING ###
    #########################################################################################################

    # Select multiple flows from the available flows dropdown
    if 'selected_flows' not in st.session_state:
        st.session_state['selected_flows'] = ['DK1->DK2']  # Default value    #show_min_flow = st.checkbox("Show Minimum Flow", value=True)
    #show_max_flow = st.checkbox("Show Maximum Flow", value=True)



    st.multiselect(
        "Select flows to plot",
        options=available_flows,
        default=st.session_state['selected_flows'],
        key='selected_flows_select',
        on_change=update_selected_flows  # Use callback to update session state
)




    if st.session_state['selected_flows']:
        for flow in st.session_state['selected_flows']:
            # Extract in_domain and out_domain from the selected flow
            in_domain, out_domain = flow.split('->')

            # Filter the DataFrame for the selected flow
            DF = get_max_flow_per_time_normalized(Border_df, in_domain, out_domain, 'flow')

            if not DF.empty:
                # Find the row with the maximum absolute Flow_FB value
              
                # Plot the flow over time
                fig = px.line(
                    DF,
                    x='dateTimeCET',  # Assuming dateTimeCET is the datetime column
                    y='maxFlow_fb',
                    title=f"Flow: {in_domain}-{out_domain} ",
                    labels={'Flow_FB': 'Flow (MW)', 'dateTimeCET': 'Datetime'}
                )
                fig.add_scatter(
                    x=DF['dateTimeCET'],  # Ensure this is the correct datetime column
                    y=DF['minFlow'],      # Ensure this column exists in the DataFrame
                    name = 'Min Flow'

                )
                fig.add_scatter(
                    x=DF['dateTimeCET'],  # Ensure this is the correct datetime column
                    y=DF['maxFlow'],      # Ensure this column exists in the DataFrame
                    name = 'Max Flow'

                )
                st.plotly_chart(fig)
            else:
                st.warning(f"No data found for the selected flow: {flow}")
    else:
        st.info("Please select at least one flow to display the plot.")

        
        
        
def update_selected_positions():
    st.session_state['selected_net_positions'] = st.session_state['net_positions_select']

# Net position plotting page
def net_position_page():
    st.title("Net Position Plots")

    # Filter Net_df to include only rows where biddingZoneFrom equals biddingZoneTo
    filtered_net_df = st.session_state['Net_df'][st.session_state['Net_df']['biddingZoneFrom'] == st.session_state['Net_df']['biddingZoneTo']]

    # Create a list of available net positions for selection
    available_net_positions = filtered_net_df['biddingZoneFrom'].unique()

    # Initialize session state for selected net positions if it doesn't exist
    if 'selected_net_positions' not in st.session_state:
        st.session_state['selected_net_positions'] = []  # Empty default value

    # Multi-select dropdown for net positions with callback
    st.multiselect(
        "Select net positions to plot",
        available_net_positions,
        default=st.session_state['selected_net_positions'],
        key='net_positions_select',
        on_change=update_selected_positions  # Callback to update session state
    )

    # If net positions are selected, plot them
    if st.session_state['selected_net_positions']:
        for position in st.session_state['selected_net_positions']:
            # Filter DataFrame for the selected net position
            position_df = filtered_net_df[filtered_net_df['biddingZoneFrom'] == position]

            if not position_df.empty:
                # Plot fref as a function of datetime for the selected net position
                fig = px.line(position_df, x='dateTimeCET', y='Flow_FB', title=f"Net Position: {position}", labels={'fref': 'Net Position (fref)', 'dateTimeCET': 'Datetime'})
                fig.add_scatter(
                    x=position_df['dateTimeCET'],  # Pass the x-axis data (datetime column)
                    y=position_df['minFlow'],      # Pass the y-axis data (minFlow column)
                    mode='lines',                  # Set the mode to lines
                    name='Min Flow'                # Label the scatter trace
)
                fig.add_scatter(
                    x=position_df['dateTimeCET'],  # Pass the x-axis data (datetime column)
                    y=position_df['maxFlow'],      # Pass the y-axis data (minFlow column)
                    mode='lines',                  # Set the mode to lines
                    name='Min Flow'                # Label the scatter trace
)
                st.plotly_chart(fig)
            else:
                st.warning(f"No data found for the selected net position: {position}")
    else:
        st.info("Please select at least one net position to display the plot.")



def shadow_price_page():
    st.title("Shadow Price Page")

    # Check if Remaining_df is in session state
    if 'Remaining_df' in st.session_state:
        Remaining_df = st.session_state['Remaining_df'].copy()

        Remaining_df.loc[Remaining_df['tso'] == 'ENERGINET', 'cnecName'] = Remaining_df['cnecName'] + " + " + Remaining_df['contName']

        
        # Summary table
        summary_table = Remaining_df.groupby('cnecName').agg(
            hours_with_shadow_price=('shadow_price', lambda x: (x > 0).sum()),
            avg_shadow_price=('shadow_price', lambda x: np.mean(x[x > 0]) if (x > 0).any() else 0),
            sum_shadow_price=('shadow_price', 'sum')
        ).reset_index()  # Ensure cnecName is not the index

        # Set dateTimeUtc as the index and display table without showing any index
        summary_table.set_index('cnecName', inplace=True)

        # Display the table with container width enabled
        st.subheader("CNEC Shadow Price Summary")
        st.dataframe(summary_table, use_container_width=True)

    else:
        st.warning("No Remaining_df found. Please go to the Main Page and initiate the API call.")


def DA_price_page():
    st.title("Day ahead Price Page")
    
    # Ensure that session state contains the DataFrames
    if 'germany_df' in st.session_state and 'other_zones_df' in st.session_state:
        germany_df = st.session_state['germany_df']
        other_zones_df = st.session_state['other_zones_df']
        
        # Sort Germany DataFrame by 'Datetime'
        germany_df = germany_df.sort_values(by='Datetime')

        # Plot Germany prices (15-minute resolution)
        st.subheader("Day-Ahead Prices for Germany (1-hour resolution)")
        germany_fig = px.line(germany_df, x='Datetime', y='Price', title="Germany Prices (1-hour resolution)", labels={'Price': 'Price (EUR/MWh)'})
        st.plotly_chart(germany_fig)

        # Plot prices for each bidding zone in other zones
        for zone in other_zones_df['Zone'].unique():
            zone_df = other_zones_df[other_zones_df['Zone'] == zone]
            # Sort each zone's DataFrame by 'Datetime'
            zone_df = zone_df.sort_values(by='Datetime')
            
            st.subheader(f"Day-Ahead Prices for {zone}")
            zone_fig = px.line(zone_df, x='Datetime', y='Price', title=f"{zone} Prices", labels={'Price': 'Price (EUR/MWh)'})
            st.plotly_chart(zone_fig)
    
    else:
        st.warning("No data available in session state. Please load data first.")



# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["API DATE SELECTOR", "CNEC PLOT PAGE", "Hourly summary - Map", "Flow plot page","Net position plots","Shadow Price","DA Price"])

if page == "API DATE SELECTOR":
     main_page()
elif page == "CNEC PLOT PAGE":
    plot_page()
elif page == "Hourly summary - Map":
    map_page()
elif page == "Flow plot page":
    flow_page()
elif page  == "Net position plots":
    net_position_page()
elif page == "Shadow Price":
    shadow_price_page() 
elif page == "DA Price":
    DA_price_page()
