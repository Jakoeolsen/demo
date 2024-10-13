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
URL1 = "https://test-publicationtool.jao.eu/nordic/api/data/finalComputation"
URL2 = "https://test-publicationtool.jao.eu/nordic/api/data/fbDomainShadowPrice"
token = "43307660-a55b-4bac-8356-0d6ea9818ad7"


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
        df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'])
    
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
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 4))
    end_date = st.date_input("End Date", value=datetime(2024, 3, 5))

    if st.button("Fetch Data"):
        
        
        df = fetch_data(start_date, end_date)
        germany_df, other_zones_df= fetch_data_prices(start_date, end_date)
        st.session_state['germany_df'] = germany_df
        st.session_state['other_zones_df'] = other_zones_df

                
        Net_df = df[df['cnecName'].str.contains("NetPosition", case=False, na=False)].copy()
        st.dataframe(df)
        
        
        
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
               
        
        ####

        # Select a TSO from Remaining_df
        tso_selected = st.selectbox("Select a TSO", Remaining_df['tso'].unique())

        # Filter the Remaining_df based on the selected TSO
        tso_filtered_df = Remaining_df[Remaining_df['tso'] == tso_selected]

        # If Energinet is chosen, combine cnecName + contName for selection
        if tso_selected == "ENERGINET":
            tso_filtered_df['cnec_cont'] = tso_filtered_df['cnecName'] + " + " + tso_filtered_df['contName']
            selected_cnec = st.selectbox("Select a CNEC", tso_filtered_df['cnec_cont'].unique())
            filtered_remaining_df = tso_filtered_df[tso_filtered_df['cnec_cont'] == selected_cnec]   ######### BUGGG/FIXED

        else:
            # Otherwise, just use cnecName
            selected_cnec = st.selectbox("Select a CNEC", tso_filtered_df['cnecName'].unique())
            filtered_remaining_df = tso_filtered_df[tso_filtered_df['cnecName'] == selected_cnec]   ######### BUGGG/FIXED


        # Filter DataFrame for the selected CNEC

        # Choose columns to plot for Remaining_df
        columns_to_plot_remaining = st.multiselect(
            "Select columns to plot for Remaining_df",
            options=['ram', 'minFlow', 'u', 'imax', 'fmax', 'frm', 'fref', 'fall', 'fnrao', 'amr', 'aac', 'iva','ram','shadow_price','Flow_FB'],
            default=['ram']
        )

        # Plot the selected CNEC for the Remaining_df
        if columns_to_plot_remaining:
            fig = px.line(filtered_remaining_df, x='dateTimeUtc', y=columns_to_plot_remaining, title=f"Plot for {selected_cnec} (Remaining_df)")
            st.plotly_chart(fig)

        # Table for Remaining_df that includes all cnecName values in the dropdown menu
        #remaining_table = tso_filtered_df[['cnecName', 'ptdf_DK1','ptdf_DK1_CO','ptdf_DK1_DE','ptdf_DK1_KS','ptdf_DK1_SK','ptdf_DK1_SB','ptdf_DK2','ptdf_DK2_KO','ptdf_DK2_SB','ptdf_FI','ptdf_FI_EL','ptdf_FI_FS','ptdf_NO1','ptdf_NO2','ptdf_NO2_ND','ptdf_NO2_SK','ptdf_NO2_NK','ptdf_NO3','ptdf_NO4','ptdf_NO5','ptdf_SE1','ptdf_SE2','ptdf_SE3','ptdf_SE3_FS','ptdf_SE3_KS','ptdf_SE3_SWL','ptdf_SE4','ptdf_SE4_BC','ptdf_SE4_NB','ptdf_SE4_SP','ptdf_SE4_SWL','ram']]  # Customize with specific columns like 'ptdf_DK1'
        #st.dataframe(remaining_table.set_index('cnecName'))

        # Section to plot CNECs associated with Border_df and Net_df
        st.subheader("Plot CNECs for Border")

        # Select a CNEC from Border_df and plot
        selected_flow_border = st.selectbox("Select a CNEC from Border_df", Border_df['cnecName'].unique())
        filtered_border_df = Border_df[Border_df['cnecName'] == selected_flow_border]

        # Choose columns to plot for Border_df
        columns_to_plot_border = st.multiselect(
            "Select columns to plot for Border_df",
            options=['ram', 'minFlow', 'u', 'imax', 'fmax', 'frm', 'fref', 'fall', 'fnrao', 'amr', 'aac', 'iva','ram','shadow_price','Flow_FB'],
            default=['ram']
        )

        if columns_to_plot_border:
            fig_border = px.line(filtered_border_df, x='dateTimeUtc', y=columns_to_plot_border, title=f"Plot for {selected_flow_border} (Border_df)")
            st.plotly_chart(fig_border)

        # Table for Border_df that includes all cnecName values in the dropdown menu
        #border_table = Border_df[['cnecName', 'ptdf_DK1','ptdf_DK1_CO','ptdf_DK1_DE','ptdf_DK1_KS','ptdf_DK1_SK','ptdf_DK1_SB','ptdf_DK2','ptdf_DK2_KO','ptdf_DK2_SB','ptdf_FI','ptdf_FI_EL','ptdf_FI_FS','ptdf_NO1','ptdf_NO2','ptdf_NO2_ND','ptdf_NO2_SK','ptdf_NO2_NK','ptdf_NO3','ptdf_NO4','ptdf_NO5','ptdf_SE1','ptdf_SE2','ptdf_SE3','ptdf_SE3_FS','ptdf_SE3_KS','ptdf_SE3_SWL','ptdf_SE4','ptdf_SE4_BC','ptdf_SE4_NB','ptdf_SE4_SP','ptdf_SE4_SWL','ram']]  # Customize with specific columns like 'ptdf_DK1'
        #st.dataframe(border_table.set_index('cnecName'))

        st.subheader("Plot CNECs for Net Position")

        # Select a CNEC from Net_df and plot
        selected_cnec_net = st.selectbox("Select a CNEC from Net_df", Net_df['cnecName'].unique())
        filtered_net_df = Net_df[Net_df['cnecName'] == selected_cnec_net]

        # Choose columns to plot for Net_df
        columns_to_plot_net = st.multiselect(
            "Select columns to plot for Net_df",
            options=['minFlow', 'fmax', 'fref', 'aac','shadow_price','Flow_FB'],
            default=['Flow_FB']
        )

        if columns_to_plot_net:
            fig_net = px.line(filtered_net_df, x='dateTimeUtc', y=columns_to_plot_net, title=f"Plot for {selected_cnec_net} (Net_df)")
            st.plotly_chart(fig_net)

        # Table for Net_df that includes all cnecName values in the dropdown menu
        #net_table = Net_df[['cnecName', 'ptdf_DK1','ptdf_DK1_CO','ptdf_DK1_DE','ptdf_DK1_KS','ptdf_DK1_SK','ptdf_DK1_SB','ptdf_DK2','ptdf_DK2_KO','ptdf_DK2_SB','ptdf_FI','ptdf_FI_EL','ptdf_FI_FS','ptdf_NO1','ptdf_NO2','ptdf_NO2_ND','ptdf_NO2_SK','ptdf_NO2_NK','ptdf_NO3','ptdf_NO4','ptdf_NO5','ptdf_SE1','ptdf_SE2','ptdf_SE3','ptdf_SE3_FS','ptdf_SE3_KS','ptdf_SE3_SWL','ptdf_SE4','ptdf_SE4_BC','ptdf_SE4_NB','ptdf_SE4_SP','ptdf_SE4_SWL','ram']]  # Customize with specific columns like 'ptdf_DK1'
        #st.dataframe(net_table.set_index('cnecName'))

    else:
        st.warning("No DataFrame found. Please go to the Main Page and initiate the API call.")


        
        
        
        
        


def map_page():
    
    ##########################################################################################
                                ### MAP###
    ###########################################################################################
    st.title("Flow Visualization Map")

    # Date selection
    Remaining_df = st.session_state.get('Remaining_df', None)

    if Remaining_df is not None:
    # Extract unique dates from Remaining_df['dateTimeUtc'] and convert them to datetime.date
        Remaining_df['date'] = pd.to_datetime(Remaining_df['dateTimeUtc']).dt.date
        unique_dates = Remaining_df['date'].unique()

    # Date picker with only available dates from Remaining_df
        selected_date = st.selectbox("Select a date", options=sorted(unique_dates))

    # Filter Remaining_df to only include rows with the selected date
        filtered_df_by_date = Remaining_df[Remaining_df['date'] == selected_date]

    # Extract unique hours for the selected date (formatted as "HH:MM")
        filtered_df_by_date['hour'] = pd.to_datetime(filtered_df_by_date['dateTimeUtc']).dt.strftime('%H:%M')
        unique_hours = filtered_df_by_date['hour'].unique()

    # Hour picker with only available hours for the selected date
        selected_hour = st.selectbox("Select an hour", options=sorted(unique_hours))
        
        

    # Combine the selected date and hour into a single datetime object
        selected_datetime = datetime.combine(selected_date, datetime.strptime(selected_hour, "%H:%M").time())

    # Format the datetime to the desired string format with +00:00
        formatted_datetime = selected_datetime.strftime("%Y-%m-%d %H:%M:%S") + "+00:00"
        st.write(f"Visualizing data for datetime: {formatted_datetime}")

    # Filter Remaining_df based on the selected date and hour
        selected_tso = st.multiselect(
            "Select TSO to table",
            options=['ENERGINET','STATNETT','FINGRID','SVK',''],
            default=['ENERGINET','STATNETT','FINGRID','SVK','']
        )
    
        tso_filtered_df = Remaining_df[Remaining_df['tso'].isin(selected_tso)]
       





    
    
        tso_filtered_df = tso_filtered_df[tso_filtered_df['dateTimeUtc'] == formatted_datetime]
 

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
    Border_df = Border_df[Border_df['dateTimeUtc'] == formatted_datetime]
    
    
    germany_df = st.session_state['germany_df']
    germany_df = germany_df[germany_df['Datetime'] == formatted_datetime]
    
    other_zones_df = st.session_state['other_zones_df']

    other_zones_df = other_zones_df[other_zones_df['Datetime'] == formatted_datetime]
    germany_df['Price'] = germany_df['Price'].round(0).astype(int)
    other_zones_df['Price'] = other_zones_df['Price'].round(0).astype(int)

    net_price = pd.concat([germany_df, other_zones_df], ignore_index=True)
    
    net_price.rename(columns={'Zone': 'region'}, inplace=True)
    net_price = net_price.set_index('region')['Price'].to_dict()
    
    
    
    
    
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
        ('DK1', 'DK2'): -50,
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
        ('NO4', 'SE2'): 80,  
        ('SE1', 'NO4'): 55,  
        ('SE1', 'SE2'): 65,  
        ('SE1', 'FI'):  40,  
        ('FI', 'SE3'):  50,  
        ('FI', 'NO4'):  45,
        ('NO2', 'NL'):  50,
        ('DK1', 'NL'):  80,
        ('NO2', 'DE'):  40, 
        ('DK1', 'DE'):  60, 
        ('DK2', 'DE'):  10, 
        ('SE4', 'PL'):  0, 
        ('SE4', 'DE'):  30,
        ('SE4', 'LI'):  30, 
        ('FI',  'ES'):  10,
    }
    flows_copy = flows.copy()

    for (zone1, zone2), value in flows_copy.items():
        max_flow_row = filter_for_zones(Border_df, zone1, zone2, 'flow')
        
        #    max_flow_row = filter_for_zones(Border_df, 'DK1', 'DK2', 'flow')    
    # If a row is returned, update the flows dictionary with the maximum absolute flow_fb value
        if max_flow_row is not None:
            flows[(max_flow_row['biddingZoneFrom'], max_flow_row['biddingZoneTo'])] = max_flow_row['Flow_FB']









    arrow_scale = 3
    fig = go.Figure()





    def add_arrow(fig, source, target, flow_value, arrow_scale):
        source_coord = np.array(COORDINATES[source])
        target_coord = np.array(COORDINATES[target])
        
        if flow_value < 0:
            source_coord, target_coord = target_coord, source_coord
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



    for (source, target), flow_value in flows.items():
        add_arrow(fig, source, target, flow_value, arrow_scale)

# Add the region markers and their values
    for region, coord in COORDINATES.items():
        max_value = max(net_price.values())
        min_value = min(net_price.values())
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
            #hovertext=f"{source} -> {target}: {flow_value:.1f}"
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

    tso_filtered_df_no_index = tso_filtered_df[['cnecName','tso','Flow_FB','shadow_price','ptdf_DK1','ptdf_DK1_CO','ptdf_DK1_DE','ptdf_DK1_KS','ptdf_DK1_SK','ptdf_DK1_SB','ptdf_DK2','ptdf_DK2_KO','ptdf_DK2_SB','ptdf_FI','ptdf_FI_EL','ptdf_FI_FS','ptdf_NO1','ptdf_NO2','ptdf_NO2_ND','ptdf_NO2_SK','ptdf_NO2_NK','ptdf_NO3','ptdf_NO4','ptdf_NO5','ptdf_SE1','ptdf_SE2','ptdf_SE3','ptdf_SE3_FS','ptdf_SE3_KS','ptdf_SE3_SWL','ptdf_SE4','ptdf_SE4_BC','ptdf_SE4_NB','ptdf_SE4_SP','ptdf_SE4_SWL','ram']].reset_index(drop=True)
    
    
    tso_filtered_df_no_index = tso_filtered_df_no_index.sort_values(by='shadow_price', ascending=False)

         
    st.dataframe(tso_filtered_df_no_index.set_index('cnecName'))

           

    
    
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
def flow_page():
    st.title("Flow Plots")

    # Define the available flows in the desired format
    available_flows = [
        'FI->FI_EL', 'DK1->DK2', 'DK1->SE3', 'DK1->NO2', 
        'DK2->SE4', 'SE4->SE3', 'SE3->NO1', 'SE3->SE2',
        'NO1->NO2', 'NO1->NO3', 'NO1->NO5', 'NO2->NO5',
        'NO5->NO3', 'NO3->SE2', 'NO3->NO4', 'NO4->SE2',
        'SE1->NO4', 'SE1->SE2', 'SE1->FI', 'FI->SE3', 
        'FI->NO4', 'NO2->NL', 'DK1->NL', 'NO2->DE', 
        'DK1->DE', 'DK2->DE', 'SE4->PL', 'SE4->DE', 
        'SE4->LI', 'FI->ES'
    ]

    # Select multiple flows
    selected_flows = st.multiselect("Select flows to plot", available_flows, default=['FI->FI_EL'])

    if selected_flows:
        for flow in selected_flows:
            # Extract in_domain and out_domain from the selected flow
            in_domain, out_domain = flow.split('->')

            # Filter the DataFrame for the selected flow (assuming fref and datetime are available in the dataset)
            filtered_df = st.session_state['Border_df'][(st.session_state['Border_df']['biddingZoneFrom'] == in_domain) &
                                                        (st.session_state['Border_df']['biddingZoneTo'] == out_domain)]

            if not filtered_df.empty:
                # Plot fref as a function of datetime for the selected flow
                fig = px.line(filtered_df, x='dateTimeUtc', y='fref', title=f"Flow: {flow}", labels={'fref': 'Flow Reference (fref)', 'dateTimeUtc': 'Datetime'})
                st.plotly_chart(fig)
            else:
                st.warning(f"No data found for the selected flow: {flow}")
    else:
        st.info("Please select at least one flow to display the plot.")
        
        
        
        
        
def net_position_page():
    st.title("Net Position Plots")

    # Filter Net_df to include only rows where biddingZoneFrom equals biddingZoneTo
    filtered_net_df = st.session_state['Net_df'][st.session_state['Net_df']['biddingZoneFrom'] == st.session_state['Net_df']['biddingZoneTo']]

    # Create a list of net positions for selection
    available_net_positions = filtered_net_df['biddingZoneFrom'].unique()

    # Multi-select dropdown for net positions
    selected_positions = st.multiselect("Select net positions to plot", available_net_positions)

    if selected_positions:
        for position in selected_positions:
            # Filter DataFrame for the selected net position
            position_df = filtered_net_df[filtered_net_df['biddingZoneFrom'] == position]

            if not position_df.empty:
                # Plot fref as a function of datetime for the selected net position
                fig = px.line(position_df, x='dateTimeUtc', y='fref', title=f"Net Position: {position}", labels={'fref': 'Net Position (fref)', 'dateTimeUtc': 'Datetime'})
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
        
        # Plot Germany prices (15-minute resolution)
        st.subheader("Day-Ahead Prices for Germany (1hour resolution)")
        germany_fig = px.scatter(germany_df, x='Datetime', y='Price', title="Germany Prices (1-hour resolution)", labels={'Price': 'Price (EUR/MWh)'})
        st.plotly_chart(germany_fig)

        # Plot prices for each bidding zone in other zones
        for zone in other_zones_df['Zone'].unique():
            zone_df = other_zones_df[other_zones_df['Zone'] == zone]
            st.subheader(f"Day-Ahead Prices for {zone}")
            zone_fig = px.scatter(zone_df, x='Datetime', y='Price', title=f"{zone} Prices", labels={'Price': 'Price (EUR/MWh)'})
            st.plotly_chart(zone_fig)
    
    else:
        st.warning("No data available in session state. Please load data first.")



# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["API DATE SELECTOR", "CNEC PLOT PAGE", "Flow Visualization Map", "Flow plot page","Net position plots","Shadow Price","DA Price"])

if page == "API DATE SELECTOR":
     main_page()
elif page == "CNEC PLOT PAGE":
    plot_page()
elif page == "Flow Visualization Map":
    map_page()
elif page == "Flow plot page":
    flow_page()
elif page  == "Net position plots":
    net_position_page()
elif page == "Shadow Price":
    shadow_price_page() 
elif page == "DA Price":
    DA_price_page()
