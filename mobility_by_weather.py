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
import plotly.express as px


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
    return f"""
    <div style='width: {width}%; background-color: #B0C4DE; height: 10px; border-radius: 5px;'></div>
    {'💨' * int(wind_speed // 5)}
    """


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
        point_row = points_df[points_df['fk_zaehler'].astype(str) == standort]

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

        if not point_row.empty:
            bezeichnung = point_row['bezeichnung'].iloc[0]
            popup = f"{bezeichnung}: {total_traffic:.0f} (Velo: {traffic_row['VELO_IN']:.0f}/{traffic_row['VELO_OUT']:.0f}, Fuss: {traffic_row['FUSS_IN']:.0f}/{traffic_row['FUSS_OUT']:.0f})"
        else:
            popup = f"Standort {standort}: {total_traffic:.0f} (Velo: {traffic_row['VELO_IN']:.0f}/{traffic_row['VELO_OUT']:.0f}, Fuss: {traffic_row['FUSS_IN']:.0f}/{traffic_row['FUSS_OUT']:.0f})"

        points.append({
            "coords": [lat, lon],
            "color": color,
            "radius": radius,
            "popup": popup
        })

    return points
def main_old():
    st.set_page_config(page_title="Weather Visualization", layout="wide")

    # Initialize session state
    if 'points' not in st.session_state:
        st.session_state.points = []
    if 'filtered_counts' not in st.session_state:
        st.session_state.filtered_counts = pd.DataFrame()

    # Load weather data
    default_path_bern = "arbeit/wetter/bern_23_clean.csv"
    default_path_zurich = "arbeit/wetter/zurich_23_clean.csv"
    wetter_Bern = load_weather_data(default_path_bern, default_path_bern)
    wetter_Zurich = load_weather_data(default_path_zurich, default_path_zurich)

    # Load mobility data
    default_points_path = "arbeit/mobility_zurich/standorte.csv"
    default_counts_path = "arbeit/mobility_zurich/zurich_mobility.csv"
    zurich_points_df = load_mobility_data(default_points_path)
    zurich_counts_df = load_mobility_data(default_counts_path)

    show_weather_rain = False
    show_weather_temp = False
    show_weather_wind = False

    cities = {
        "Bern": [46.9480, 7.4474],
        "Zurich": [47.3769, 8.5417]
    }
    left, b, middle, c, right = st.columns([2, 1, 10, 1, 2])

    with left:
        st.header("Controls")
        city = st.selectbox("City", ["Bern", "Zurich", "both"])
        start_date = st.date_input("Start date", value=datetime(2023, 1, 1))
        start_datetime = datetime(start_date.year, start_date.month, start_date.day, 1)  # Set to 01:00 UTC
        start_timestamp = int(start_datetime.timestamp())
        st.write(f"Timestamp: {start_timestamp}")
        unit = st.selectbox("Unit", ["Hours", "Days", "Months"])
        duration = st.slider("Duration", 1, 24 if unit == "Hours" else 31, 12 if unit == "Hours" else 10)

        show_dataf = st.toggle("show data")
        show_weather = st.toggle("show weather average")
        if st.toggle("show weather detail", True):
            show_weather_rain = st.toggle("show rain")
            show_weather_temp = st.toggle("show temp")
            show_weather_wind = st.toggle("show wind")

        # Select and filter data
        wetter = None
        if city == "Bern" and wetter_Bern is not None:
            wetter = wetter_Bern
        elif city == "Zurich" and wetter_Zurich is not None:
            wetter = wetter_Zurich
        elif city == "both" and wetter_Bern is not None and wetter_Zurich is not None:
            wetter = pd.concat([wetter_Bern, wetter_Zurich])

        if wetter is None:
            st.error(f"No data available for {city}. Check CSV files.")
            filtered_df = pd.DataFrame()
        else:
            filtered_df = filter_weather_data(wetter, start_timestamp, duration, unit)
            # Update points and filtered counts
            st.session_state.points = []
            st.session_state.filtered_counts = pd.DataFrame()
            if city in ["Zurich", "both"] and zurich_points_df is not None and zurich_counts_df is not None:
                # Filter counts for session state
                counts_df = zurich_counts_df.copy()
                counts_df['DATUM'] = pd.to_datetime(counts_df['DATUM']).apply(lambda x: int(x.timestamp()))
                if unit == "Hours":
                    end_timestamp = start_timestamp + duration * 3600
                elif unit == "Days":
                    end_timestamp = start_timestamp + duration * 86400
                else:  # Months
                    start_date = datetime.fromtimestamp(start_timestamp)
                    end_date = start_date + pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(duration)
                    end_timestamp = int(end_date.timestamp())
                st.session_state.filtered_counts = counts_df[(counts_df['DATUM'] >= start_timestamp) & (counts_df['DATUM'] <= end_timestamp)]
                st.session_state.points = process_traffic_data(zurich_points_df, zurich_counts_df, start_timestamp, duration, unit)
            elif city == "Bern":
                city_coords = cities["Bern"]
                st.session_state.points = [{
                    "coords": [city_coords[0], city_coords[1]],
                    "color": "grey",
                    "radius": 3,
                    "popup": "Bern: No traffic data"
                }]

    with middle:
        st.header("wie beeinflusst das Wetter die Nutzung von Verkehrsmitteln in Bern und Zürich?")
        st.write("in dieser Datenvisualisierung kann für das Jahr 2023 Wetter und Mobilitätsdaten verglichen werden.")
        st.write("dieser Prototyp ist ein zwischenstand, es werden weitere Daten, Visualisierungen und (statistische) Auswertungen hinzugefügt.")
        if not filtered_df.empty:
            representative_df = get_representative_weather(filtered_df, duration, unit)
            if not representative_df.empty:
                st.subheader("Weather Conditions")
                cols = st.columns(len(representative_df))
                for i, (col, row) in enumerate(zip(cols, representative_df.iterrows())):
                    with col:
                        timestamp = pd.to_datetime(row[1]['dt'], unit='s')
                        if unit == "Hours":
                            label = timestamp.strftime('%H:%M')
                        elif unit == "Days":
                            label = timestamp.strftime('%d.%m.')
                        else:  # Months
                            label = timestamp.strftime('%b')
                        st.write(label)
                        if show_weather:
                            emoji = get_weather_emoji(row[1]['weather_icon'])
                            if st.button(f"{emoji}", key=f"{emoji}_{i}"):
                                st.write(
                                    f"{emoji} {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {row[1]['weather_description'].capitalize()} "
                                    f"(Temp: {row[1]['temp']:.1f}°C, Humidity: {row[1]['humidity']}%, Wind: {row[1]['wind_speed']:.1f} m/s)")
                        if show_weather_wind:
                            wind_speed = row[1]['wind_speed']
                            st.markdown(f"Wind: {wind_speed:.1f} m/s {wind_visual(wind_speed)}", unsafe_allow_html=True)

                        if show_weather_rain:
                            rain = row[1].get('rain_1h', 0)
                            if rain >= 0:
                                st.markdown(f"{rain} mm 🌧️ {rain_bar(rain)}", unsafe_allow_html=True)
                            else:
                                st.markdown(f"trocken {rain_bar(rain)}", unsafe_allow_html=True)

                        if show_weather_temp:
                            temp = row[1]['temp']
                            st.markdown(
                                f"<div style='{temp_to_color(temp)}; padding: 5px; border-radius: 5px;'>Temp: {temp:.1f}°C 🌡️</div>",
                                unsafe_allow_html=True
                            )

        else:
            st.warning("No weather data available. Check time range or CSV data.")
        st.subheader("karte")
        if not filtered_df.empty:
            if city == "both":
                center = [(cities["Bern"][0] + cities["Zurich"][0]) / 2,
                          (cities["Bern"][1] + cities["Zurich"][1]) / 2]
                m = display_map(center, st.session_state.points, zoom=9)
                folium.Marker(cities["Bern"], popup="Bern").add_to(m)
                folium.Marker(cities["Zurich"], popup="Zurich").add_to(m)
            else:
                m = display_map(cities[city], st.session_state.points)
            st.components.v1.html(m._repr_html_(), height=600)
        else:
            st.warning("Map not displayed due to missing weather data.")
        if show_dataf:
            st.subheader("Data Table")
            st.dataframe(filtered_df[['dt', 'dt_iso', 'temp', 'humidity', 'wind_speed', 'weather_description']],
                         use_container_width=True)
            if zurich_points_df is not None:
                st.dataframe(zurich_points_df, use_container_width=True)
            if zurich_counts_df is not None:
                st.dataframe(zurich_counts_df, use_container_width=True)

    with right:
        st.header("Key Statistics")
        if city in ["Zurich", "both"] and not st.session_state.filtered_counts.empty:
            filtered_counts = st.session_state.filtered_counts
            # Calculate totals
            total_pedestrians = filtered_counts['FUSS_IN'].fillna(0).sum() + filtered_counts['FUSS_OUT'].fillna(0).sum()
            total_cyclists = filtered_counts['VELO_IN'].fillna(0).sum() + filtered_counts['VELO_OUT'].fillna(0).sum()

            # Calculate timespan in hours
            timespan_hours = (end_timestamp - start_timestamp) / 3600
            avg_pedestrians = total_pedestrians / timespan_hours if timespan_hours > 0 else 0
            avg_cyclists = total_cyclists / timespan_hours if timespan_hours > 0 else 0

            st.markdown(f"**Total Pedestrians**: {total_pedestrians:,.0f}")
            st.markdown(f"**Total Cyclists**: {total_cyclists:,.0f}")
            st.markdown(f"**Avg. Pedestrians/Hour**: {avg_pedestrians:,.1f}")
            st.markdown(f"**Avg. Cyclists/Hour**: {avg_cyclists:,.1f}")
            st.markdown(f"**Filtered Rows**: {len(filtered_counts)} (debug)")
        else:
            st.markdown("No mobility data available for the selected city or time range.")


def main():
    st.set_page_config(page_title="Weather Visualization", layout="wide")

    # Initialize session state
    if 'points' not in st.session_state:
        st.session_state.points = []
    if 'filtered_counts' not in st.session_state:
        st.session_state.filtered_counts = pd.DataFrame()

    # Load weather data
    default_path_bern = "arbeit/wetter/bern_23_clean.csv"
    default_path_zurich = "arbeit/wetter/zurich_23_clean.csv"
    wetter_Bern = load_weather_data(default_path_bern, default_path_bern)
    wetter_Zurich = load_weather_data(default_path_zurich, default_path_zurich)

    # Load mobility data
    default_points_path = "arbeit/mobility_zurich/standorte.csv"
    default_counts_path = "arbeit/mobility_zurich/zurich_mobility.csv"
    zurich_points_df = load_mobility_data(default_points_path)
    zurich_counts_df = load_mobility_data(default_counts_path)

    show_weather_rain = False
    show_weather_temp = False
    show_weather_wind = False

    cities = {
        "Bern": [46.9480, 7.4474],
        "Zurich": [47.3769, 8.5417]
    }
    left, b, middle, c, right = st.columns([2, 1, 10, 1, 2])

    with left:
        st.header("Controls")
        city = st.selectbox("City", ["Bern", "Zurich", "both"])
        start_date = st.date_input("Start date", value=datetime(2023, 1, 1))
        start_datetime = datetime(start_date.year, start_date.month, start_date.day, 1)  # Set to 01:00 UTC
        start_timestamp = int(start_datetime.timestamp())
        st.write(f"Timestamp: {start_timestamp}")
        unit = st.selectbox("Unit", ["Hours", "Days", "Months"])
        duration = st.slider("Duration", 1, 24 if unit == "Hours" else 31, 12 if unit == "Hours" else 10)

        show_dataf = st.toggle("show data")
        show_weather = st.toggle("show weather average")
        if st.toggle("show weather detail", True):
            show_weather_rain = st.toggle("show rain")
            show_weather_temp = st.toggle("show temp")
            show_weather_wind = st.toggle("show wind")

        # Select and filter data
        wetter = None
        if city == "Bern" and wetter_Bern is not None:
            wetter = wetter_Bern
        elif city == "Zurich" and wetter_Zurich is not None:
            wetter = wetter_Zurich
        elif city == "both" and wetter_Bern is not None and wetter_Zurich is not None:
            wetter = pd.concat([wetter_Bern, wetter_Zurich])

        if wetter is None:
            st.error(f"No data available for {city}. Check CSV files.")
            filtered_df = pd.DataFrame()
        else:
            filtered_df = filter_weather_data(wetter, start_timestamp, duration, unit)
            # Update points and filtered counts
            st.session_state.points = []
            st.session_state.filtered_counts = pd.DataFrame()
            if city in ["Zurich", "both"] and zurich_points_df is not None and zurich_counts_df is not None:
                # Filter counts for session state
                counts_df = zurich_counts_df.copy()
                counts_df['DATUM'] = pd.to_datetime(counts_df['DATUM']).apply(lambda x: int(x.timestamp()))
                if unit == "Hours":
                    end_timestamp = start_timestamp + duration * 3600
                elif unit == "Days":
                    end_timestamp = start_timestamp + duration * 86400
                else:  # Months
                    start_date = datetime.fromtimestamp(start_timestamp)
                    end_date = start_date + pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(duration)
                    end_timestamp = int(end_date.timestamp())
                st.session_state.filtered_counts = counts_df[(counts_df['DATUM'] >= start_timestamp) & (counts_df['DATUM'] <= end_timestamp)]
                st.session_state.points = process_traffic_data(zurich_points_df, zurich_counts_df, start_timestamp, duration, unit)
            elif city == "Bern":
                city_coords = cities["Bern"]
                st.session_state.points = [{
                    "coords": [city_coords[0], city_coords[1]],
                    "color": "grey",
                    "radius": 3,
                    "popup": "Bern: No traffic data"
                }]

    with middle:
        st.header("wie beeinflusst das Wetter die Nutzung von Verkehrsmitteln in Bern und Zürich?")
        st.write("in dieser Datenvisualisierung kann für das Jahr 2023 Wetter und Mobilitätsdaten verglichen werden.")
        st.write("dieser Prototyp ist ein zwischenstand, es werden weitere Daten, Visualisierungen und (statistische) Auswertungen hinzugefügt.")
        if not filtered_df.empty:
            representative_df = get_representative_weather(filtered_df, duration, unit)
            if not representative_df.empty:
                st.subheader("Weather Conditions")
                cols = st.columns(len(representative_df))
                for i, (col, row) in enumerate(zip(cols, representative_df.iterrows())):
                    with col:
                        timestamp = pd.to_datetime(row[1]['dt'], unit='s')
                        if unit == "Hours":
                            label = timestamp.strftime('%H:%M')
                        elif unit == "Days":
                            label = timestamp.strftime('%d.%m.')
                        else:  # Months
                            label = timestamp.strftime('%b')
                        st.write(label)
                        if show_weather:
                            emoji = get_weather_emoji(row[1]['weather_icon'])
                            if st.button(f"{emoji}", key=f"{emoji}_{i}"):
                                st.write(
                                    f"{emoji} {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {row[1]['weather_description'].capitalize()} "
                                    f"(Temp: {row[1]['temp']:.1f}°C, Humidity: {row[1]['humidity']}%, Wind: {row[1]['wind_speed']:.1f} m/s)")
                        if show_weather_wind:
                            wind_speed = row[1]['wind_speed']
                            st.markdown(f"Wind: {wind_speed:.1f} m/s {wind_visual(wind_speed)}", unsafe_allow_html=True)

                        if show_weather_rain:
                            rain = row[1].get('rain_1h', 0)
                            if rain >= 0:
                                st.markdown(f"{rain} mm 🌧️ {rain_bar(rain)}", unsafe_allow_html=True)
                            else:
                                st.markdown(f"trocken {rain_bar(rain)}", unsafe_allow_html=True)

                        if show_weather_temp:
                            temp = row[1]['temp']
                            st.markdown(
                                f"<div style='{temp_to_color(temp)}; padding: 5px; border-radius: 5px;'>Temp: {temp:.1f}°C 🌡️</div>",
                                unsafe_allow_html=True
                            )

        else:
            st.warning("No weather data available. Check time range or CSV data.")
        st.subheader("karte")
        if not filtered_df.empty:
            if city == "both":
                center = [(cities["Bern"][0] + cities["Zurich"][0]) / 2,
                          (cities["Bern"][1] + cities["Zurich"][1]) / 2]
                m = display_map(center, st.session_state.points, zoom=9)
                folium.Marker(cities["Bern"], popup="Bern").add_to(m)
                folium.Marker(cities["Zurich"], popup="Zurich").add_to(m)
            else:
                m = display_map(cities[city], st.session_state.points)
            st.components.v1.html(m._repr_html_(), height=600)
        else:
            st.warning("Map not displayed due to missing weather data.")
        if show_dataf:
            st.subheader("Data Table")
            st.dataframe(filtered_df[['dt', 'dt_iso', 'temp', 'humidity', 'wind_speed', 'weather_description']],
                         use_container_width=True)
            if zurich_points_df is not None:
                st.dataframe(zurich_points_df, use_container_width=True)
            if zurich_counts_df is not None:
                st.dataframe(zurich_counts_df, use_container_width=True)


    with right:
        st.header("Key Statistics")
        if city in ["Zurich", "both"] and not st.session_state.filtered_counts.empty:
            filtered_counts = st.session_state.filtered_counts
            # Calculate totals for selected period
            total_pedestrians = filtered_counts['FUSS_IN'].fillna(0).sum() + filtered_counts['FUSS_OUT'].fillna(0).sum()
            total_cyclists = filtered_counts['VELO_IN'].fillna(0).sum() + filtered_counts['VELO_OUT'].fillna(0).sum()

            # Calculate timespan in hours
            if unit == "Hours":
                end_timestamp = start_timestamp + duration * 3600
            elif unit == "Days":
                end_timestamp = start_timestamp + duration * 86400
            else:  # Months
                start_date = datetime.fromtimestamp(start_timestamp)
                end_date = start_date + pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(duration)
                end_timestamp = int(end_date.timestamp())
            timespan_hours = (end_timestamp - start_timestamp) / 3600
            avg_pedestrians = total_pedestrians / timespan_hours if timespan_hours > 0 else 0
            avg_cyclists = total_cyclists / timespan_hours if timespan_hours > 0 else 0

            st.markdown(f"**Total Pedestrians**: {total_pedestrians:,.0f}")
            st.markdown(f"**Total Cyclists**: {total_cyclists:,.0f}")
            st.markdown(f"**Avg. Pedestrians/Hour**: {avg_pedestrians:,.1f}")
            st.markdown(f"**Avg. Cyclists/Hour**: {avg_cyclists:,.1f}")

            # Calculate yearly averages
            if zurich_counts_df is not None:
                year_counts = zurich_counts_df.copy()
                total_year_peds = year_counts['FUSS_IN'].fillna(0).sum() + year_counts['FUSS_OUT'].fillna(0).sum()
                total_year_cycs = year_counts['VELO_IN'].fillna(0).sum() + year_counts['VELO_OUT'].fillna(0).sum()
                year_hours = 365 * 24  # Approximate hours in 2023
                avg_year_peds = total_year_peds / year_hours
                avg_year_cycs = total_year_cycs / year_hours
                st.markdown(f"**Yearly Avg. Pedestrians/Hour**: {avg_year_peds:,.1f}")
                st.markdown(f"**Yearly Avg. Cyclists/Hour**: {avg_year_cycs:,.1f}")

            # Calculate time-of-day averages
            avg_time_peds = 0
            avg_time_cycs = 0
            comparison_text = "No comparison available due to missing time-of-day data."
            if zurich_counts_df is not None:
                time_counts = zurich_counts_df.copy()
                time_counts['DATUM_DT'] = pd.to_datetime(time_counts['DATUM'])
                if unit == "Hours":
                    time_counts['time_key'] = time_counts['DATUM_DT'].dt.hour
                    start_hour = start_datetime.hour
                    time_avg = time_counts[time_counts['time_key'] == start_hour].groupby('time_key').agg({
                        'FUSS_IN': 'sum', 'FUSS_OUT': 'sum', 'VELO_IN': 'sum', 'VELO_OUT': 'sum'
                    })
                else:  # Days or Months, use daily averages
                    time_counts['time_key'] = time_counts['DATUM_DT'].dt.date
                    time_avg = time_counts.groupby('time_key').agg({
                        'FUSS_IN': 'sum', 'FUSS_OUT': 'sum', 'VELO_IN': 'sum', 'VELO_OUT': 'sum'
                    })
                    time_avg = time_avg.mean().to_frame().T  # Average across all days

                if not time_avg.empty:
                    time_avg['peds'] = time_avg['FUSS_IN'].fillna(0) + time_avg['FUSS_OUT'].fillna(0)
                    time_avg['cycs'] = time_avg['VELO_IN'].fillna(0) + time_avg['VELO_OUT'].fillna(0)
                    avg_time_peds = time_avg['peds'].iloc[0] / (1 if unit == "Hours" else 24)
                    avg_time_cycs = time_avg['cycs'].iloc[0] / (1 if unit == "Hours" else 24)

                    # Comparison sentence
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
        else:
            st.markdown("No mobility data available for the selected city or time range.")

if __name__ == "__main__":
    main()
