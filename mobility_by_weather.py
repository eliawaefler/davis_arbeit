"""
CAS Datenvisualisierung und Statistik Arbeit von Elia Wäfler
proudly created with help from GROK.
"""



import pandas as pd
import streamlit as st
import os
from datetime import datetime, timedelta
import numpy as np
import folium
from pyproj import Transformer
import plotly.graph_objects as go


def swiss_to_wgs84(easting, northing):
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return [lat, lon]


def display_map(city_coords, points, zoom=13):
    m = folium.Map(
        location=city_coords,
        zoom_start=zoom,
        tiles="cartodbpositron",
        attr="",
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=False
    )
    for point in points:
        if isinstance(point, dict) and "coords" in point and "color" in point and "radius" in point:
            folium.CircleMarker(
                location=point["coords"],
                radius=point["radius"],
                color=point["color"],
                fill=True,
                fill_color=point["color"],
                popup=point.get("popup", "")
            ).add_to(m)
    return m


def load_weather_data(file, file_path=None):
    try:
        if isinstance(file, str) and os.path.exists(file):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file)
        if df.empty or 'dt' not in df.columns:
            st.error(f"No valid data or missing dt in {file_path or 'uploaded file'}.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading {file_path or 'uploaded file'}: {e}")
        return None


def load_mobility_data(file_path):
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if df.empty:
                st.error(f"No data in {file_path}.")
                return None
            return df
        else:
            st.error(f"File {file_path} does not exist.")
            return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None


def is_dark(timestamp: int) -> bool:
    # Convert timestamp to Zurich datetime (UTC+1, no DST adjustment here)
    dt = datetime.fromtimestamp(timestamp + 3600*2)  # Add 1 hour for CET

    # Get month and hour
    month = dt.month
    hour = dt.hour

    # Define sunrise/sunset times (approximate for Zurich)
    sunrise = 8 if month in [11, 12, 1, 2] else 7 if month in [3, 4, 9, 10] else 6
    sunset = 18 if month in [11, 12, 1, 2] else 19 if month in [3, 4, 9, 10] else 21

    # Check if time is before sunrise or after sunset
    return hour < sunrise or hour >= sunset


def filter_weather_data(df, start_timestamp, duration, unit):
    if df is not None and not df.empty:
        if unit == "Hours":
            end_timestamp = start_timestamp + duration * 3600
        elif unit == "Days":
            end_timestamp = start_timestamp + duration * 86400
        else:  # Months
            start_date = datetime.fromtimestamp(start_timestamp)
            end_date = (start_date + pd.offsets.MonthBegin(duration)).timestamp()
            end_timestamp = int(end_date)
        filtered = df[(df['dt'] >= start_timestamp) & (df['dt'] <= end_timestamp)]
        return filtered
    return pd.DataFrame()


def get_representative_weather(df, duration, unit):
    if df.empty:
        return df
    if unit == "Months":
        df['date'] = pd.to_datetime(df['dt'], unit='s').dt.date
        daily = df.groupby('date').agg({
            'temp': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'weather_icon': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            'weather_description': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            'dt': 'first'
        }).reset_index()
        if len(daily) > duration:
            indices = np.linspace(0, len(daily) - 1, duration, dtype=int)
            daily = daily.iloc[indices]
        return daily
    elif unit == "Days":
        df['date'] = pd.to_datetime(df['dt'], unit='s').dt.date
        daily = df.groupby('date').agg({
            'temp': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'weather_icon': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            'weather_description': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            'dt': 'first'
        }).reset_index()
        if len(daily) > duration:
            indices = np.linspace(0, len(daily) - 1, duration, dtype=int)
            daily = daily.iloc[indices]
        return daily
    else:  # Hours
        if len(df) > duration:
            indices = np.linspace(0, len(df) - 1, duration, dtype=int)
            df = df.iloc[indices]
        return df


def get_weather_emoji(weather_icon):
    emoji_map = {
        '01d': ':sunny:', '01n': ':star2:', '02d': ':sun_small_cloud:', '02n': ':stars:',
        '03d': ':mostly_sunny:', '03n': ':mostly_sunny:', '04d': ':sun_behind_cloud:', '04n': ':sun_behind_cloud:',
        '09d': ':rain_cloud:', '09n': ':rain_cloud:', '10d': ':partly_sunny_rain:', '10n': ':sun_behind_rain_cloud:',
        '11d': ':lightning_cloud:', '11n': ':lightning_cloud:', '13d': ':snow_cloud:', '13n': ':snow_cloud:',
        '50d': ':fog:', '50n': ':fog:'
    }
    return emoji_map.get(weather_icon, ':cloud:')


def temp_to_color(temp):
    if temp < 0:
        return "background-color: #ADD8E6"
    elif temp < 10:
        return "background-color: #90EE90"
    elif temp < 20:
        return "background-color: #FFFFE0"
    else:
        return "background-color: #FF6347"


def rain_bar(rain):
    if rain >= 0.01:
        max_rain = 1.5
        width = min(rain / max_rain * 100, 100)
        return f"""
        <div style='width: {width}%; background-color: #1E90FF; height: 10px; border-radius: 5px;'></div>
        """
    else:
        return f"""
        <div style='width: 0%; background-color: #1E90FF; height: 10px; border-radius: 5px;'></div>
        """


def wind_visual(wind_speed):
    max_wind = 20
    width = min(wind_speed / max_wind * 100, 100)
    n_clouds = int(wind_speed // 5)
    clouds = {'💨' * n_clouds}
    return f"""
    <div style='width: {width}%; background-color: #B0C4DE; height: 10px; border-radius: 5px;'></div>          
    """

def get_name(points_df, fk_number):
    #st.dataframe(points_df)
    for i, p in points_df.iterrows():
        if str(p[4]) == str(fk_number):
            return str(p[1])

def process_traffic_data(points_df, counts_df, start_timestamp, duration, unit):
    if points_df is None or counts_df is None:
        return []

    # Convert DATUM to timestamp
    counts_df['DATUM'] = pd.to_datetime(counts_df['DATUM']).apply(lambda x: int(x.timestamp()))

    # Filter counts by time range
    if unit == "Hours":
        end_timestamp = start_timestamp + duration * 3600
    elif unit == "Days":
        end_timestamp = start_timestamp + duration * 86400
    else:  # Months
        start_date = datetime.fromtimestamp(start_timestamp)
        end_date = (start_date + pd.offsets.MonthBegin(duration)).timestamp()
        end_timestamp = int(end_date)

    filtered_counts = counts_df[(counts_df['DATUM'] >= start_timestamp) & (counts_df['DATUM'] <= end_timestamp)]

    # Aggregate traffic by FK_STANDORT
    traffic = filtered_counts.groupby('FK_STANDORT').agg({
        'VELO_IN': 'sum',
        'VELO_OUT': 'sum',
        'FUSS_IN': 'sum',
        'FUSS_OUT': 'sum',
        'OST': 'first',  # Take first coordinate for consistency
        'NORD': 'first'
    }).reset_index()

    traffic['total_traffic'] = (traffic['VELO_IN'].fillna(0) +
                               traffic['VELO_OUT'].fillna(0) +
                               traffic['FUSS_IN'].fillna(0) +
                               traffic['FUSS_OUT'].fillna(0))

    # Normalize traffic for scaling (0 to 1)
    max_traffic = traffic['total_traffic'].max() if traffic['total_traffic'].max() > 0 else 1
    traffic['traffic_norm'] = traffic['total_traffic'] / max_traffic

    points = []
    for _, traffic_row in traffic.iterrows():
        easting, northing = traffic_row['OST'], traffic_row['NORD']
        lat, lon = swiss_to_wgs84(easting, northing)

        # Match FK_STANDORT with fk_zaehler for popup name
        standort = str(traffic_row['FK_STANDORT'])
        #point_row = points_df[points_df['fk_zaehler'].astype(str) == standort]

        bezeichnung = get_name(points_df, str(int(standort[:-2])))

        total_traffic = traffic_row['total_traffic']
        traffic_norm = traffic_row['traffic_norm']

        if total_traffic == 0:
            color = 'grey'
            radius = 3
        else:
            # Interpolate color from blue (low) to red (high)
            blue = int(255 * (1 - traffic_norm))
            red = int(255 * traffic_norm)
            color = f'rgb({red}, 0, {blue})'
            # Scale radius from 5 to 15
            radius = 5 + 10 * traffic_norm

        popup = f"Standort {bezeichnung}: {total_traffic:.0f} (Velo: {traffic_row['VELO_IN']:.0f}/{traffic_row['VELO_OUT']:.0f}, Fuss: {traffic_row['FUSS_IN']:.0f}/{traffic_row['FUSS_OUT']:.0f})"

        points.append({
            "coords": [lat, lon],
            "color": color,
            "radius": radius,
            "popup": popup
        })

    return points

def initialize_session_state():
    if 'points' not in st.session_state:
        st.session_state.points = []
    if 'filtered_counts' not in st.session_state:
        st.session_state.filtered_counts = pd.DataFrame()

def load_data():
    default_path_zurich = "zurich_wetter.csv"
    wetter_Zurich = load_weather_data(default_path_zurich, default_path_zurich)
    default_points_path = "zurich_standorte.csv"
    zurich_points_df = load_mobility_data(default_points_path)
    zurich_counts_df_1 = load_mobility_data("zurich_mobility_1.csv")
    zurich_counts_df_2 = load_mobility_data("zurich_mobility_2.csv")
    zurich_counts_df_3 = load_mobility_data("zurich_mobility_3.csv")
    zurich_counts_df = pd.concat([zurich_counts_df_1, zurich_counts_df_2, zurich_counts_df_3], ignore_index=True)
    return wetter_Zurich, zurich_points_df, zurich_counts_df

def setup_layout():
    st.set_page_config(page_title="Weather Visualization", layout="wide")
    return st.columns([3, 1, 20, 1, 1])

def create_controls(left):
    with left:
        st.subheader("Controls")
        city = "Zurich"
        start_date = st.date_input("Start date", value=datetime(2023, 1, 1))
        start_datetime = datetime(start_date.year, start_date.month, start_date.day, 1)
        start_timestamp = int(start_datetime.timestamp())
        unit = "Hours"
        duration = st.select_slider('Zeitraum', range(0, 24), value=(13, 14))
        start_timestamp += duration[0]*3600+3600
        duration = int(duration[1]-duration[0]+1)
        show_abstract = st.toggle("Wetter Interpretation anzeigen")
        show_weather = show_weather_rain = show_weather_temp = show_weather_wind = False
        if show_abstract:
            show_weather = st.toggle("Wetter Interpretation gesamt")
            if st.toggle("Wetter Interpretation Detail", True):
                show_weather_rain = st.toggle("Regen")
                show_weather_temp = st.toggle("Temperatur")
                show_weather_wind = st.toggle("Wind")
        show_map = st.toggle("Karte anzeigen")
        show_line = st.toggle("Linienchart anzeigen")
        show_dataf = st.toggle("Datensatz anzeigen")

        return city, start_timestamp, unit, duration, show_dataf, show_abstract, show_weather, show_weather_rain, show_weather_temp, show_weather_wind, show_map, show_line

def filter_data(city, wetter_Zurich, zurich_points_df, zurich_counts_df, start_timestamp, duration, unit):
    wetter = wetter_Zurich if city == "Zurich" and wetter_Zurich is not None else pd.DataFrame()
    if wetter.empty:
        st.error(f"No data available for {city}. Check CSV files.")
        filtered_df = pd.DataFrame()
    else:
        filtered_df = filter_weather_data(wetter, start_timestamp, duration, unit)
        st.session_state.points = []
        st.session_state.filtered_counts = pd.DataFrame()
        if city in ["Zurich", "both"] and zurich_points_df is not None and zurich_counts_df is not None:
            counts_df = zurich_counts_df.copy()
            counts_df['DATUM'] = pd.to_datetime(counts_df['DATUM']).apply(lambda x: int(x.timestamp()))
            end_timestamp = start_timestamp + duration * 3600
            st.session_state.filtered_counts = counts_df[(counts_df['DATUM'] >= start_timestamp) & (counts_df['DATUM'] <= end_timestamp)]
            st.session_state.points = process_traffic_data(zurich_points_df, zurich_counts_df, start_timestamp, duration, unit)
        elif city == "Bern":
            city_coords = [46.9480, 7.4474]
            st.session_state.points = [{"coords": [city_coords[0], city_coords[1]], "color": "grey", "radius": 3, "popup": "Bern: No traffic data"}]
    return filtered_df

def create_line_chart_(filtered_df, filtered_counts, start_timestamp, duration):
    if filtered_df.empty or filtered_counts.empty:
        return
    df = filtered_df.copy()
    df['datetime'] = pd.to_datetime(df['dt'], unit='s')
    counts_df = filtered_counts.copy()
    counts_df['datetime'] = pd.to_datetime(counts_df['DATUM'], unit='s')
    counts_df['cyclists'] = counts_df['VELO_IN'].fillna(0) + counts_df['VELO_OUT'].fillna(0)
    counts_df['pedestrians'] = counts_df['FUSS_IN'].fillna(0) + counts_df['FUSS_OUT'].fillna(0)
    hourly_counts = counts_df.groupby(counts_df['datetime'].dt.floor('H')).agg({
        'cyclists': 'sum',
        'pedestrians': 'sum'
    }).reset_index()
    df = df.set_index('datetime').resample('H').mean(numeric_only=True).reset_index()
    merged_df = pd.merge_asof(
        df.sort_values('datetime'),
        hourly_counts.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )
    merged_df['hour'] = merged_df['datetime'].dt.strftime('%H:%M')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged_df['hour'],
        y=merged_df['temp'],
        name='Temperature (°C)',
        yaxis='y1',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=merged_df['hour'],
        y=merged_df['cyclists'],
        name='Cyclists',
        yaxis='y2',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=merged_df['hour'],
        y=merged_df['pedestrians'],
        name='Pedestrians',
        yaxis='y2',
        line=dict(color='green')
    ))
    fig.update_layout(
        title='Temperature, Cyclists, and Pedestrians Over Time',
        xaxis=dict(title='Hour'),
        yaxis=dict(
            title=dict(text='Temperature (°C)', font=dict(color='red')),
            tickfont=dict(color='red')
        ),
        yaxis2=dict(
            title=dict(text='Count', font=dict(color='blue')),
            tickfont=dict(color='blue'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)


def create_line_chart(filtered_df, filtered_counts, start_timestamp, duration):
    if filtered_df.empty or filtered_counts.empty:
        return
    df = filtered_df.copy()
    df['datetime'] = pd.to_datetime(df['dt'], unit='s')
    counts_df = filtered_counts.copy()
    counts_df['datetime'] = pd.to_datetime(counts_df['DATUM'], unit='s')
    counts_df['cyclists'] = counts_df['VELO_IN'].fillna(0) + counts_df['VELO_OUT'].fillna(0)
    counts_df['pedestrians'] = counts_df['FUSS_IN'].fillna(0) + counts_df['FUSS_OUT'].fillna(0)
    hourly_counts = counts_df.groupby(counts_df['datetime'].dt.floor('H')).agg({
        'cyclists': 'sum',
        'pedestrians': 'sum'
    }).reset_index()
    df = df.set_index('datetime').resample('H').mean(numeric_only=True).reset_index()
    merged_df = pd.merge_asof(
        df.sort_values('datetime'),
        hourly_counts.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )
    merged_df['hour'] = merged_df['datetime'].dt.strftime('%H:%M')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged_df['hour'][:-1],
        y=merged_df['temp'][:-1],
        name='Temperature (°C)',
        yaxis='y1',
        line=dict(color='red'),
        mode='lines',
        connectgaps=False
    ))
    fig.add_trace(go.Scatter(
        x=merged_df['hour'][:-1],
        y=merged_df['cyclists'][:-1],
        name='Cyclists',
        yaxis='y2',
        line=dict(color='blue'),
        mode='lines',
        connectgaps=False
    ))
    fig.add_trace(go.Scatter(
        x=merged_df['hour'][:-1],
        y=merged_df['pedestrians'][:-1],
        name='Pedestrians',
        yaxis='y2',
        line=dict(color='green'),
        mode='lines',
        connectgaps=False
    ))

    fig.update_layout(
        title='Vergleich von Temperatur (°C) zu Anzahl Fussgänger und Anzahl Fahrradfahrern',
        xaxis=dict(
            title='Hour',
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text='Temperature (°C)', font=dict(color='red')),
            tickfont=dict(color='red')
        ),
        yaxis2=dict(
            title=dict(text='Count', font=dict(color='blue')),
            tickfont=dict(color='blue'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        height=600)

    st.plotly_chart(fig, use_container_width=True)


def display_middle(middle, filtered_df, duration, start_timestamp, unit, show_abstract, show_weather, show_weather_rain,
                   show_weather_temp, show_weather_wind, show_map, show_line, show_dataf, city, zurich_points_df, zurich_counts_df):
    cities = {"Bern": [46.9480, 7.4474], "Zurich": [47.3769, 8.5417]}
    with middle:
        st.header("Anzahl Fussgänger und Fahrradfahrer in Zürich 2023")
        st.write("In dieser Datenvisualisierung können für das Jahr 2023 Wetter und Mobilitätsdaten verglichen werden.")
        if not filtered_df.empty:
            representative_df = get_representative_weather(filtered_df, duration, unit)
            if show_abstract and not representative_df.empty:
                st.subheader("Wetter")
                st.write("Interpretation des Wetters mit icons")
                cols = st.columns(len(representative_df))
                for i, (col, row) in enumerate(zip(cols, representative_df.iterrows())):
                    with col:
                        timestamp = pd.to_datetime(row[1]['dt'], unit='s')
                        label = timestamp.strftime('%H:%M' if unit == "Hours" else '%d.%m.' if unit == "Days" else '%b')
                        if unit == "Hours" and i == len(cols)-1:
                            timestamp = pd.to_datetime(row[1]['dt']-3600, unit='s')
                        st.write(label)
                        if show_weather:
                            emoji = get_weather_emoji(row[1]['weather_icon'])
                            button_key = f"{emoji}_{i}"
                            st.markdown(
                                f"""
                                <style>
                                .rectangle{'' if is_dark(int(row[1]['dt'])-4*3600) else '1'} {{
                                    width: 100px;
                                    height: 5px;
                                    background-color: {'#333333' if is_dark(int(row[1]['dt'])-4*3600) else '#ffffed'};
                                    border-radius: 0;
                                }}
                                </style>
                                <div class="rectangle{'' if is_dark(int(row[1]['dt'])-4*3600) else '1'}"></div>
                                """,
                                unsafe_allow_html=True
                            )
                            if st.button(f"{emoji}", key=button_key):
                                timestamp = datetime.fromtimestamp(int(row[1]['dt']) + 3600)
                                st.write(
                                    f"{emoji} {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {row[1]['weather_description'].capitalize()} "
                                    f"(Temp: {row[1]['temp']:.1f}°C, Humidity: {row[1]['humidity']}%, Wind: {row[1]['wind_speed']:.1f} m/s)"
                                )
                        if show_weather_wind:
                            wind_speed = row[1]['wind_speed']
                            st.markdown(f"Wind: {wind_speed:.1f} m/s {wind_visual(wind_speed)}", unsafe_allow_html=True)
                        if show_weather_rain:
                            rain = row[1].get('rain_1h', 0)
                            st.markdown(f"{rain} mm 🌧️ {rain_bar(rain)}" if rain >= 0 else f"trocken {rain_bar(rain)}", unsafe_allow_html=True)
                        if show_weather_temp:
                            temp = row[1]['temp']
                            st.markdown(
                                f"<div style='{temp_to_color(temp)}; padding: 5px; border-radius: 5px;'>Temp: {temp:.1f}°C 🌡️</div>",
                                unsafe_allow_html=True
                            )
            if show_line:
                st.subheader("Vergleich von Temperatur (°C) zu Anzahl Fussgänger und Anzahl Fahrradfahrern")
                create_line_chart(filtered_df, st.session_state.filtered_counts, start_timestamp, duration)
            if show_map and not filtered_df.empty:
                st.subheader("Karte")
                if city == "both":
                    center = [(cities["Bern"][0] + cities["Zurich"][0]) / 2, (cities["Bern"][1] + cities["Zurich"][1]) / 2]
                    m = display_map(center, st.session_state.points, zoom=9)
                    folium.Marker(cities["Bern"], popup="Bern").add_to(m)
                    folium.Marker(cities["Zurich"], popup="Zurich").add_to(m)
                else:
                    m = display_map(cities[city], st.session_state.points)
                st.components.v1.html(m._repr_html_(), height=600)
                st.write("auf dieser Karte sieht man die Messstationen in Zürich. die Grösse der Kreise zeigt"
                         "die Anzahl FussgängerInnen oder FahrradfahrerInnen, die im festgelegten Zeitraum erkannt wurden.")
            elif show_map:
                st.warning("Map not displayed due to missing weather data.")
            if show_dataf:
                st.subheader("Data Table")
                st.write("Wetter")
                st.dataframe(filtered_df[['dt', 'dt_iso', 'temp', 'humidity', 'wind_speed', 'weather_description']], use_container_width=True)
                st.write("Wetterinformationen von OpenWeatherMap History Bulk! (gekauft)")
                st.write(".csv pro Stadt (Zürich) und PRo Jahr (2023). Pro Stunde eine Zeile.")
                st.write("")
                st.write("Standorte")
                if zurich_points_df is not None:
                    st.dataframe(zurich_points_df, use_container_width=True)
                st.write("Die Standorte der Messstationen von OpenData Zürich. csv imt einer Zeile pro Standort."
                         "URL: https://www.stadt-zuerich.ch/geodaten/download/Standorte_der_automatischen_Fuss__und_Velozaehlungen?format=10008")
                st.write("")
                st.write("Fussgänger und Fahrradfahrer")
                if zurich_counts_df is not None:
                    st.dataframe(zurich_counts_df, use_container_width=True)
                st.write("Fussgänger und Fahrradfahrer datenset der Stadt Zürich. csv mit Zielen pro Stunde und Messstation."
                         "https://data.stadt-zuerich.ch/dataset/ted_taz_verkehrszaehlungen_werte_fussgaenger_velo/download/2023_verkehrszaehlungen_werte_fussgaenger_velo.csv")

        else:
            st.warning("No weather data available. Check time range or CSV data.")



def display_statistics(right, city, start_timestamp, duration, unit, zurich_counts_df, show_dataf):
    with right:
        st.header("Key Statistics")
        if city in ["Zurich", "both"] and not st.session_state.filtered_counts.empty:
            filtered_counts = st.session_state.filtered_counts
            total_pedestrians = filtered_counts['FUSS_IN'].fillna(0).sum() + filtered_counts['FUSS_OUT'].fillna(0).sum()
            total_cyclists = filtered_counts['VELO_IN'].fillna(0).sum() + filtered_counts['VELO_OUT'].fillna(0).sum()
            end_timestamp = start_timestamp + duration * 3600
            timespan_hours = (end_timestamp - start_timestamp) / 3600
            avg_pedestrians = total_pedestrians / timespan_hours if timespan_hours > 0 else 0
            avg_cyclists = total_cyclists / timespan_hours if timespan_hours > 0 else 0

            st.markdown(f"**Total Pedestrians**: {total_pedestrians:,.0f}")
            st.markdown(f"**Total Cyclists**: {total_cyclists:,.0f}")
            st.markdown(f"**Avg. Pedestrians/Hour**: {avg_pedestrians:,.1f}")
            st.markdown(f"**Avg. Cyclists/Hour**: {avg_cyclists:,.1f}")

            if zurich_counts_df is not None:
                year_counts = zurich_counts_df.copy()
                year_counts['DATUM_DT'] = pd.to_datetime(year_counts['DATUM'])
                year_counts['HOUR'] = year_counts['DATUM_DT'].dt.hour
                start_hour = start_timestamp
                end_hour = start_timestamp + duration
                yearly_time_counts = year_counts[year_counts['HOUR'].between(start_hour, end_hour)]
                total_year_peds = yearly_time_counts['FUSS_IN'].fillna(0).sum() + yearly_time_counts['FUSS_OUT'].fillna(0).sum()
                total_year_cycs = yearly_time_counts['VELO_IN'].fillna(0).sum() + yearly_time_counts['VELO_OUT'].fillna(0).sum()
                days_in_year = 365
                hours_per_day = (end_hour - start_hour + 1)
                total_hours = days_in_year * hours_per_day
                avg_year_peds = total_year_peds / total_hours if total_hours > 0 else 0
                avg_year_cycs = total_year_cycs / total_hours if total_hours > 0 else 0
                st.markdown(f"**Yearly Avg. Pedestrians/Hour**: {avg_year_peds:,.1f}")
                st.markdown(f"**Yearly Avg. Cyclists/Hour**: {avg_year_cycs:,.1f}")

            avg_time_peds = avg_time_cycs = 0
            comparison_text = "No comparison available due to missing time-of-day data."
            if zurich_counts_df is not None:
                time_counts = zurich_counts_df.copy()
                time_counts['DATUM_DT'] = pd.to_datetime(time_counts['DATUM'])
                time_counts['time_key'] = time_counts['DATUM_DT'].dt.hour
                start_hour = datetime.fromtimestamp(start_timestamp).hour
                time_avg = time_counts[time_counts['time_key'] == start_hour].groupby('time_key').agg({
                    'FUSS_IN': 'sum', 'FUSS_OUT': 'sum', 'VELO_IN': 'sum', 'VELO_OUT': 'sum'
                })
                if not time_avg.empty:
                    time_avg['peds'] = time_avg['FUSS_IN'].fillna(0) + time_avg['FUSS_OUT'].fillna(0)
                    time_avg['cycs'] = time_avg['VELO_IN'].fillna(0) + time_avg['VELO_OUT'].fillna(0)
                    avg_time_peds = time_avg['peds'].iloc[0]
                    avg_time_cycs = time_avg['cycs'].iloc[0]
                    cycs_diff = ((avg_cyclists - avg_time_cycs) / avg_time_cycs * 100) if avg_time_cycs > 0 else 0
                    peds_diff = ((avg_pedestrians - avg_time_peds) / avg_time_peds * 100) if avg_time_peds > 0 else 0
                    cycs_comp = "more" if cycs_diff >= 0 else "less"
                    peds_comp = "more" if peds_diff >= 0 else "less"
                    comparison_text = (
                        f"In your selected time period, {total_cyclists:,.0f} cyclists and {total_pedestrians:,.0f} pedestrians were registered. "
                        f"This is {abs(cycs_diff):.1f}% {cycs_comp} cyclists and {abs(peds_diff):.1f}% {peds_comp} pedestrians than the average for this time of day."
                    )
                else:
                    st.markdown("**Debug**: No time-of-day data available for the selected period.")
                st.markdown(f"**Time-of-Day Avg. Pedestrians/Hour**: {avg_time_peds:,.1f}")
                st.markdown(f"**Time-of-Day Avg. Cyclists/Hour**: {avg_time_cycs:,.1f}")
                st.markdown(comparison_text)

            if show_dataf:
                pass
                """
                st.subheader("Average Pedestrians and Cyclists by Hour of Day")
                st.dataframe(hourly_avg_df.style.format({
                    'PEDESTRIANS': '{:,.1f}',
                    'CYCLISTS': '{:,.1f}',
                    'HOUR': '{:d}'
                }), use_container_width=True)
                """
        else:
            st.markdown("No mobility data available for the selected city or time range.")

def main():
    initialize_session_state()
    wetter_Zurich, zurich_points_df, zurich_counts_df = load_data()
    left, _, middle, _, right = setup_layout()
    city, start_timestamp, unit, int_duration, show_dataf, show_abstract, show_weather, show_weather_rain, show_weather_temp, show_weather_wind, show_map, show_line = create_controls(left)
    filtered_df = filter_data(city, wetter_Zurich, zurich_points_df, zurich_counts_df, start_timestamp, int_duration, unit)
    display_middle(middle, filtered_df, int_duration, start_timestamp, unit, show_abstract, show_weather, show_weather_rain, show_weather_temp, show_weather_wind, show_map, show_line, show_dataf, city, zurich_points_df, zurich_counts_df)
    #display_right(right, filtered_df, int_duration, start_timestamp, unit, show_abstract, show_weather, show_weather_rain, show_weather_temp, show_weather_wind, show_map, show_line, show_dataf, city, zurich_points_df, zurich_counts_df)
    #display_statistics(right, city, start_timestamp, int_duration, unit, zurich_counts_df, show_dataf)
    # Hardcoded hourly averages DataFrame
    hourly_avg_df = pd.DataFrame({
        'HOUR': list(range(24)),
        'PEDESTRIANS': [
            33.082877, 18.028082, 12.176511, 8.149315, 7.815753, 13.467123, 39.016438, 96.151370,
            133.253425, 129.134932, 155.736986, 211.797945, 279.871233, 267.652055, 268.368493,
            273.584932, 290.453425, 309.326027, 268.673973, 184.763699, 131.844521, 105.811644,
            91.679452, 62.532877
        ],
        'CYCLISTS': [
            89.842466, 48.247945, 30.392857, 20.201370, 18.921918, 47.866438, 180.135616, 507.241096,
            696.415753, 402.367808, 325.613014, 374.406849, 418.848630, 429.432192, 396.541781,
            425.765068, 543.806849, 810.130137, 787.923288, 516.699315, 354.093836, 274.828082,
            232.952055, 161.209589
        ]
    })

if __name__ == '__main__':
    main()
