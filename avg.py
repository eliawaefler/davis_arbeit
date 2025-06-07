import pandas as pd
import streamlit as st
import os

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


def calc_hourly_averages():
    # Load and concatenate data
    zurich_counts_df_1 = load_mobility_data("zurich_mobility_1.csv")
    zurich_counts_df_2 = load_mobility_data("zurich_mobility_2.csv")
    zurich_counts_df_3 = load_mobility_data("zurich_mobility_3.csv")
    zurich_counts_df = pd.concat([zurich_counts_df_1, zurich_counts_df_2], ignore_index=True)
    df = pd.concat([zurich_counts_df, zurich_counts_df_3], ignore_index=True)

    if df is None:
        return None

    # Convert DATUM to datetime
    df['DATUM'] = pd.to_datetime(df['DATUM'])

    # Extract hour from DATUM
    df['HOUR'] = df['DATUM'].dt.hour

    # Replace NaN with 0
    df[['VELO_IN', 'VELO_OUT', 'FUSS_IN', 'FUSS_OUT']] = df[['VELO_IN', 'VELO_OUT', 'FUSS_IN', 'FUSS_OUT']].fillna(0)

    # Sum across all FK_STANDORT for each hour of each day
    daily_hourly_sums = df.groupby(['DATUM', 'HOUR'])[
        ['FUSS_IN', 'FUSS_OUT', 'VELO_IN', 'VELO_OUT']].sum().reset_index()

    # Calculate total pedestrians and cyclists
    daily_hourly_sums['PEDESTRIANS'] = daily_hourly_sums['FUSS_IN'] + daily_hourly_sums['FUSS_OUT']
    daily_hourly_sums['CYCLISTS'] = daily_hourly_sums['VELO_IN'] + daily_hourly_sums['VELO_OUT']

    # Average across all days for each hour
    hourly_avg = daily_hourly_sums.groupby('HOUR')[['PEDESTRIANS', 'CYCLISTS']].mean().reset_index()

    return hourly_avg

print(calc_hourly_averages())